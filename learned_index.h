//
// Created by Ben on 17/10/2019.
//

#ifndef LEARNED_INDEX_LEARNED_INDEX_H
#define LEARNED_INDEX_LEARNED_INDEX_H

#include <vector>
#include <cassert>
#include "models/nn_model.h"
#include "models/btree_model.h"
#include "records.h"


namespace LearnedIndex {
    template<typename T, typename Index> //todo: assert arithmetic types
    class RecursiveModelIndex {
        using nn_model_t = LearnedIndex::NNModel<T, Index>;

        //todo: copy ctor, operator=

    public:
        RecursiveModelIndex(const std::vector<int> &stage_sizes, const std::vector<std::vector<int>>& hidden_layers_sizes,
                            const std::vector<std::vector<int>>& hidden_layers_widths, const std::vector<T>& indexed_data /* todo: move to training */);

        /* fit the model to our data (might replace some of the last stage models with BTrees) */
        void hybrid_end_to_end_training(Index btree_err_threshold = 0 /*todo: remove 0 default*/);

        Index get_index(T val) const;

    private:
        std::vector<std::vector<std::unique_ptr<LearnedIndex::Model<T, Index>>>> staged_models;
        std::vector<std::pair<Index, Index>> under_and_over_estimation_err;
        const std::vector<T>& indexed_data;

        void create_models(const std::vector<int> &stage_sizes, const std::vector<std::vector<int>> &hidden_layers_sizes,
                           const std::vector<std::vector<int>> &hidden_layers_widths);

        /**
         *
         * @param val - the val we'd like to locate in the indexed_data
         * @param predicted_index - the index of val as predicted by the learned index.
         * @return the exact index of val in indexed_data
         *
         * @pre 0 <= predicted_index < |indexed_data|
         */
        inline Index exponential_search(T val, Index predicted_index) const noexcept;

        //binary search val in the range [data_start, data_end)
        inline Index binary_search(Index data_start, Index data_end, T val) const noexcept;
    };

    template<typename T, typename Index>
    RecursiveModelIndex<T, Index>::RecursiveModelIndex(const std::vector<int> &stage_sizes,
                                                       const std::vector<std::vector<int>> &hidden_layers_sizes,
                                                       const std::vector<std::vector<int>> &hidden_layers_widths,
                                                       const std::vector<T> &indexed_data):
            indexed_data(indexed_data){ //TODO: parse a config file NN DEPTH WIDTH, EQUI-DEPTH BUCKETS, ...

        assert(!stage_sizes.empty());
        assert(stage_sizes.at(0) == 1);

        for(int i=0; i<stage_sizes.back(); i++){
            under_and_over_estimation_err.emplace_back(0, 0);
        }

        create_models(stage_sizes, hidden_layers_sizes, hidden_layers_widths);
        hybrid_end_to_end_training();
    }

    template<typename T, typename Index>
    void RecursiveModelIndex<T, Index>::hybrid_end_to_end_training(Index btree_err_threshold) { //todo: dont use in ctor
        std::vector<std::vector<LearnedIndex::Records<T, Index>>> tmp_records;
        tmp_records.resize(staged_models.size());
        for(int i=0; i<staged_models.size(); i++){
            tmp_records.at(i).resize(staged_models.at(i).size());
        }

        tmp_records.at(0).at(0) = LearnedIndex::Records<T, Index>(indexed_data);

        const int n_model_stages = staged_models.size();
        for(auto stage_num = 0; stage_num < n_model_stages; stage_num++){
            const int models_in_stage = staged_models.at(stage_num).size();
            for(auto model_num = 0; model_num < models_in_stage; model_num++){
                if(tmp_records.at(stage_num).at(model_num).empty()){
                    continue;
                }
                staged_models.at(stage_num).at(model_num)->train(tmp_records.at(stage_num).at(model_num).inputs(),
                                                                 tmp_records.at(stage_num).at(model_num).desired_outputs());
                if(stage_num < n_model_stages - 1){
                    for(auto r = 0; r < tmp_records.at(stage_num).at(model_num).size(); r++){
                        const Index predicted_index = staged_models.at(stage_num).at(model_num)->predict(tmp_records.at(stage_num).at(model_num).inputs().at(r));
                        Index next_stage_model_idx = staged_models.at(stage_num + 1).size() * (predicted_index / indexed_data.size()); //todo: devide by N-1 not N (same in nn_model)
                        next_stage_model_idx = std::min(next_stage_model_idx, staged_models.at(stage_num + 1).size() - 1);
                        tmp_records.at(stage_num + 1).at(next_stage_model_idx).insert(tmp_records.at(stage_num).at(model_num).get_record(r));
                    }
                    //reclaim memory
                    tmp_records.at(stage_num).at(model_num).clear();
                }
            }
        }

        const auto last_stage_records_vec = tmp_records.back();
        for(auto model_num = 0; model_num < staged_models.back().size(); model_num++){
            const auto model_records = last_stage_records_vec.at(model_num);
            for(auto r = 0; r < model_records.size(); r++){
                T input = model_records.inputs().at(r);
                Index predicted_index = staged_models.back().at(model_num)->predict(input);
                Index correct_index = model_records.desired_outputs().at(r);
                //TODO: REFACTOR TO METHOD UPDATE_EST_ERR:
                Index err = 0;
                if(predicted_index > correct_index){
                    err = predicted_index - correct_index;
                    if(err > under_and_over_estimation_err.at(model_num).second){
                        under_and_over_estimation_err.at(model_num).second = err;
                    }
                } else{
                    err = correct_index - predicted_index;
                    if(err < under_and_over_estimation_err.at(model_num).first){
                        under_and_over_estimation_err.at(model_num).first = err;
                    }
                }
                std::cout << "err: " << err << std::endl;
                if(err > btree_err_threshold){
                    auto replacement_btree_ptr = std::make_unique<LearnedIndex::BTreeModel<T, Index>>();
                    replacement_btree_ptr->train(model_records.inputs(), model_records.desired_outputs());
                    staged_models.back().at(model_num) = std::move(replacement_btree_ptr);
                    under_and_over_estimation_err.at(model_num) = std::make_pair(0, 0);
                    break;
                }
            }
        }
    }

    template<typename T, typename Index>
    Index RecursiveModelIndex<T, Index>::get_index(T val) const {

        unsigned int next_model_idx = 0;
        for(auto stage_num = 0; stage_num < staged_models.size(); stage_num++) {
            const Index model_prediction = staged_models[stage_num][next_model_idx]->predict(val);
            if(stage_num != staged_models.size() - 1){
                const auto predicted_next_model_idx = staged_models[stage_num + 1].size() * (model_prediction / indexed_data.size()); //todo: devide by N-1 not N (same in nn_model)
                next_model_idx = std::min(predicted_next_model_idx, staged_models[stage_num + 1].size() - 1);
            } else{
                return exponential_search(val, model_prediction); //todo: use under_over_err!!!
            }

        }

    }

    template<typename T, typename Index>
    void RecursiveModelIndex<T, Index>::create_models(const std::vector<int> &stage_sizes,
                                                      const std::vector<std::vector<int>> &hidden_layers_sizes,
                                                      const std::vector<std::vector<int>> &hidden_layers_widths) {
        const auto number_of_stages = stage_sizes.size();
        staged_models.resize(number_of_stages);
        for (std::vector<int>::size_type stage_num = 0; stage_num < stage_sizes.size(); stage_num++) {
            for (int model_num = 0; model_num < stage_sizes.at(stage_num); model_num++) {
                //we always start with NNs in all stages (the model is linear in the case that #hidden_layers==0)
                staged_models.at(stage_num).push_back(
                        std::make_unique<nn_model_t>(hidden_layers_sizes.at(stage_num).at(model_num), hidden_layers_widths.at(stage_num).at(model_num))
                );
            }
        }
    }

    template<typename T, typename Index>
    Index RecursiveModelIndex<T, Index>::exponential_search(T val, Index predicted_index) const noexcept {
        Index N = indexed_data.size();
        assert(N != 0);

        Index bound = predicted_index;

        if(indexed_data[bound] < val){
            while(bound < N && indexed_data[bound] < val){
                bound *= 2;
            }
            return binary_search(bound / 2, bound, val);
        } else{ // indexed_data[bound] >= val
            while(bound > 0 && indexed_data[bound] >= val){
                bound /= 2;
            }
            return binary_search(bound, bound * 2, val);
        }
    }

    template<typename T, typename Index>
    Index RecursiveModelIndex<T, Index>::binary_search(Index data_start, Index data_end, T val) const noexcept {
        auto result_iter = std::lower_bound(&indexed_data[data_start], &indexed_data[data_end], val);
        assert(result_iter != indexed_data.end());
        return *result_iter;
    }


}

#endif //LEARNED_INDEX_LEARNED_INDEX_H
