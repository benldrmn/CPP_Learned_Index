//
// Created by Ben on 21/10/2019.
//

#ifndef LEARNED_INDEX_RECORDS_H
#define LEARNED_INDEX_RECORDS_H

#include <vector>
#include <cassert>
#include <memory> //unique_ptr
#include <numeric> //iota

namespace LearnedIndex {
    template<typename T, typename Index>
    class Records{

    public:

        explicit Records(const std::vector<T>& indexed_data);

        Records() = default;

        //todo: operator=, copy ctor, move ctor

        void insert(const T input, const Index desired_output);

        void insert(const std::pair<T, Index>& p);

        std::size_t size() const noexcept;

        bool empty() const noexcept;

        const std::vector<T>& inputs() const noexcept;

        const std::vector<Index>& desired_outputs() const noexcept;

        std::pair<T, Index> get_record(std::size_t r) const noexcept;

        void clear() noexcept;

    private:
        std::vector<T> m_inputs;
        std::vector<Index> m_desired_outputs;
    };

    template<typename T, typename Index>
    Records<T, Index>::Records(const std::vector<T> &indexed_data): m_inputs(std::vector<T>(indexed_data)){
        m_desired_outputs = std::vector<Index>(indexed_data.size());
        std::iota(m_desired_outputs.begin(), m_desired_outputs.end(), 0); //fill [0, indexed_data.size() - 1]
    }

    template<typename T, typename Index>
    void Records<T, Index>::insert(const T input, const Index desired_output) {
        m_inputs.push_back(input);
        m_desired_outputs.push_back(desired_output);
    }

    template<typename T, typename Index>
    void Records<T, Index>::insert(const std::pair<T, Index> &p) {
        return insert(p.first, p.second);
    }

    template<typename T, typename Index>
    std::size_t Records<T, Index>::size() const noexcept {
        return m_inputs.size();
    }

    template<typename T, typename Index>
    bool Records<T, Index>::empty() const noexcept {
        return m_inputs.empty();
    }

    template<typename T, typename Index>
    const std::vector<T> &Records<T, Index>::inputs() const noexcept {
        return m_inputs;
    }

    template<typename T, typename Index>
    const std::vector<Index> &Records<T, Index>::desired_outputs() const noexcept {
        return m_desired_outputs;
    }

    template<typename T, typename Index>
    std::pair<T, Index> Records<T, Index>::get_record(std::size_t r) const noexcept {
        assert(r >= 0);
        assert(r < inputs.size());
        return std::make_pair(m_inputs[r], m_desired_outputs[r]);
    }

    template<typename T, typename Index>
    void Records<T, Index>::clear() noexcept {
        m_inputs.clear();
        m_desired_outputs.clear();
    }
}


#endif //LEARNED_INDEX_RECORDS_H
