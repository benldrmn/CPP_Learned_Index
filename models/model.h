//
// Created by Ben on 17/10/2019.
//

#ifndef LEARNED_INDEX_MODEL_H
#define LEARNED_INDEX_MODEL_H

#include <vector>

namespace LearnedIndex{
    template<typename T, typename Index>
    class Model {
    public:
        virtual ~Model() = default;
        virtual void train(const std::vector<T>& inputs, const std::vector<Index>& desired_outputs) = 0;
        //return the predicted 0 <= index <= max index (== indexed_data_size - 1)
        virtual Index predict(const T input) = 0;
    };
}

#endif //LEARNED_INDEX_MODEL_H
