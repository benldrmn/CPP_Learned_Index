//
// Created by Ben on 18/10/2019.
//

#ifndef LEARNED_INDEX_NN_MODEL_H
#define LEARNED_INDEX_NN_MODEL_H

#include <thread>
#include <algorithm>
#include <numeric>
#include "../tiny_dnn/tiny_dnn/tiny_dnn.h"
#include "model.h"

namespace LearnedIndex{
    template<typename T, typename Index>
    class NNModel: public Model<T, Index>{

        static constexpr int BATCH_SIZE = 128;
        static constexpr int EPOCHS_PER_TRAIN = 3;

        public:
            NNModel(int hidden_layers_num, int hidden_layers_width);

            ~NNModel() override = default;

        void train(const std::vector<T>& inputs, const std::vector<Index>& desired_outputs) override;


        virtual Index predict(const T input);

        private:
            tiny_dnn::network<tiny_dnn::sequential> neural_net;
            std::size_t indexed_data_size;
            tiny_dnn::float_t mean;
            tiny_dnn::float_t standard_deviation;
            tiny_dnn::float_t data_min;
            tiny_dnn::float_t data_max;



            inline tiny_dnn::float_t calculate_mean(const std::vector<T>& inputs) const noexcept;

            tiny_dnn::float_t calculate_standard_deviation(const std::vector<T>& inputs, tiny_dnn::float_t mean_) const;

            inline tiny_dnn::float_t normalize(T input) const noexcept;

            inline tiny_dnn::float_t clip(tiny_dnn::float_t val, tiny_dnn::float_t low, tiny_dnn::float_t high) noexcept;
        };

    template<typename T, typename Index>
    NNModel<T, Index>::NNModel(int hidden_layers_num, int hidden_layers_width):
            indexed_data_size(0), mean(0), standard_deviation(1), data_min(0), data_max(0){
        if(hidden_layers_num == 0){
            neural_net << tiny_dnn::fully_connected_layer(1, 1); // a simple linear model
            return;
        }

        neural_net << tiny_dnn::fully_connected_layer(1, hidden_layers_width) << tiny_dnn::activation::relu();
        int hidden_layers_left = hidden_layers_num - 1;
        while(hidden_layers_left > 0){
            neural_net << tiny_dnn::fully_connected_layer(hidden_layers_width, hidden_layers_width) << tiny_dnn::activation::relu();
            hidden_layers_left--;
        }
        neural_net << tiny_dnn::fully_connected_layer(hidden_layers_width, 1);
    }

    template<typename T, typename Index>
    void NNModel<T, Index>::train(const std::vector<T> &inputs, const std::vector<Index> &desired_outputs) {
        assert(!inputs.empty()); //todo: throw
        assert(inputs.size() == desired_outputs.size());

        indexed_data_size = inputs.size();
        mean = calculate_mean(inputs);
        standard_deviation = calculate_standard_deviation(inputs, mean);
        auto x = std::min_element(inputs.begin(), inputs.end());
        auto y = std::max_element(inputs.begin(), inputs.end());
        data_min = static_cast<tiny_dnn::float_t>(*std::min_element(inputs.begin(), inputs.end()));
        data_max = static_cast<tiny_dnn::float_t>(*std::max_element(inputs.begin(), inputs.end()));

        const auto n_inputs = inputs.size();
        auto n_threads = std::thread::hardware_concurrency();
        if(!n_threads){
            n_threads = CNN_TASK_SIZE;
        }
        std::vector<tiny_dnn::vec_t> normalized_inputs;
        normalized_inputs.reserve(n_inputs);
        std::for_each(inputs.begin(), inputs.end(),
                      [&](const T& t) {normalized_inputs.emplace_back(tiny_dnn::vec_t(1, normalize(t)));});

        std::vector<tiny_dnn::vec_t> outputs;
        outputs.reserve(n_inputs);
        const tiny_dnn::float_t N = indexed_data_size;
        std::for_each(desired_outputs.begin(), desired_outputs.end(),
                      [&outputs, &N](const T& output){outputs.emplace_back(1, static_cast<tiny_dnn::float_t>(output) / N);});
        //todo: ^ divide by N-1?

        auto batch_size = std::min(static_cast<std::size_t>(BATCH_SIZE), outputs.size());
        tiny_dnn::gradient_descent opt; //todo: move EPOCHS to be an argument, also lr(alpha)?
        neural_net.fit<tiny_dnn::mse>(opt, normalized_inputs, outputs, batch_size, EPOCHS_PER_TRAIN,
                                      []() {}, []() {}, false, n_threads);
    }

    template<typename T, typename Index>
    Index NNModel<T, Index>::predict(const T input) {
        const tiny_dnn::float_t out = neural_net.predict(tiny_dnn::vec_t(1, normalize(input)))[0];
        return static_cast<Index>(clip(out * indexed_data_size, 0, indexed_data_size - 1));
    }

    template<typename T, typename Index>
    tiny_dnn::float_t NNModel<T, Index>::calculate_mean(const std::vector<T> &inputs) const noexcept {
        const auto sum = static_cast<tiny_dnn::float_t>(std::accumulate(inputs.begin(), inputs.end(), 0.0));
        return sum / static_cast<tiny_dnn::float_t>(inputs.size());
    }

    template<typename T, typename Index>
    tiny_dnn::float_t
    NNModel<T, Index>::calculate_standard_deviation(const std::vector<T> &inputs, tiny_dnn::float_t mean_) const {
        std::vector<tiny_dnn::float_t> diff(inputs.size());
        std::transform(inputs.begin(), inputs.end(), diff.begin(), [mean_](tiny_dnn::float_t x) { return x - mean_; });
        tiny_dnn::float_t squares_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        auto stdev = std::sqrt(squares_sum / static_cast<tiny_dnn::float_t>(inputs.size()));
        return stdev == 0? 1 : stdev;
    }

    template<typename T, typename Index>
    tiny_dnn::float_t NNModel<T, Index>::normalize(T input) const noexcept {
        //assert(standard_deviation != 0);
        //return (static_cast<tiny_dnn::float_t>(input) - mean) / standard_deviation;
        const auto div = (data_max - data_min) == 0? 1 : data_max - data_min; //todo: optimize this branch away
        return (static_cast<tiny_dnn::float_t>(input) - data_min)/div;
    }

    template<typename T, typename Index>
    tiny_dnn::float_t NNModel<T, Index>::clip(tiny_dnn::float_t val, tiny_dnn::float_t low, tiny_dnn::float_t high) noexcept {
        assert(low <= high);
        return val < low? low : ( high < val? high : val );
    }


}

#endif //LEARNED_INDEX_NN_MODEL_H
