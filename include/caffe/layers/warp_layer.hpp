#ifndef CAFFE_WARP_LAYER_HPP_
#define CAFFE_WARP_LAYER_HPP_

#include <vector>
#include "caffe/layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class WarpLayer : public Layer<Dtype>
{
  public:
    explicit WarpLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
    
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const {return "Bias";}

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    
  private:
    int num;

};  //class WarpLayer


}  // namespace caffe


#endif  // CAFFE_WARP_LAYER_HPP_
