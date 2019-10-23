//
// Created by Ben on 20/10/2019.
//

#ifndef LEARNED_INDEX_BTREE_MODEL_H
#define LEARNED_INDEX_BTREE_MODEL_H

#include "model.h"
#include "../btree/btree_map.h"

namespace LearnedIndex {
    template<typename T, typename Index>
    class BTreeModel : public Model<T, Index> {

    public:
        BTreeModel() = default; //todo: copy ctor, move ctor, operator=
        ~BTreeModel() override = default;

        void train(const std::vector<T>& inputs, const std::vector<Index>& desired_outputs) override;

        Index predict(const T input) override;

    private:
        btree::btree_map<T, Index> btree;
    };

    template<typename T, typename Index>
    void BTreeModel<T, Index>::train(const std::vector<T> &inputs, const std::vector<Index> &desired_outputs) {
        assert(inputs.size() == desired_outputs.size());
        for(std::size_t i = 0; i < inputs.size(); i++){
            btree.insert(std::make_pair(desired_outputs[i], inputs[i]));
        }
    }

    template<typename T, typename Index>
    Index BTreeModel<T, Index>::predict(const T input) {
        for(auto iter = btree.begin(); iter != btree.end(); iter++){
            std::cout << iter->first << " " << iter->second << std::endl;
        }
        assert(btree.find(input) != btree.end());

        return btree.find(input)->second;
    }
};
#endif //LEARNED_INDEX_BTREE_MODEL_H
