#include <vector>

#include "caffe/layers/warp_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void WarpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void WarpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{}

template <typename Dtype>
void WarpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{}

template <typename Dtype>
void WarpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{}

template <typename Dtype>
void WarpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{}




//INSTANTIATE_CLASS(WarpLayer);
//REGISTER_LAYER_CLASS(Warp);


} // namespace caffe


