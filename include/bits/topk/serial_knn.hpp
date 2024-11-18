#ifndef BITS_TOPK_SERIAL_KNN_HPP_
#define BITS_TOPK_SERIAL_KNN_HPP_

#include <string>
#include <vector>

#include "bits/knn.hpp"

/** CPU baseline which sorts the distances using std::sort
 */
class serial_knn : public knn
{
public:
    std::string id() const override { return "serial"; }

    void prepare() override;
    void distances() override;
    void selection() override;
    std::vector<distance_pair> finish() override;

private:
    std::vector<distance_pair> dist_;
};

#endif // BITS_TOPK_SERIAL_KNN_HPP_
