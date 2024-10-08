#ifndef SERIAL_KNN_HPP_
#define SERIAL_KNN_HPP_

#include "bits/knn.hpp"

/** CPU baseline which sorts the distances using std::sort
 */
class serial_knn : public knn
{
public:
    inline std::string id() const override { return "serial"; }

    void prepare() override;
    void distances() override;
    void selection() override;
    std::vector<distance_pair> finish() override;

private:
    std::vector<distance_pair> dist_;
};

#endif // SERIAL_KNN_HPP_
