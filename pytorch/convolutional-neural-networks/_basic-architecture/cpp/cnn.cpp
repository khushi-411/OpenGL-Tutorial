// https://pytorch.org/cppdocs/frontend.html
// https://github.com/krshrimali/Digit-Recognition-MNIST-SVHN-PyTorch-CPP/blob/master/training.cpp

#include <torch\torch.h>

struct Net : torch::nn::Module {
	Net() {
		// initialization
		conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5)));
		conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5)));
		//conv2_drop = register_module("conv2_drop", torch::nn::FeatureAlphaDropout());
		fc1 = register_module("fc1", torch::nn::Linear(320, 50));
		fc2 = register_module("fc2", torch::nn::Linear(50, 10));
	}
	// implementation
	torch::Tensor forward(torch::Tensor x) {
		// tensor.size(0) ---- one-dimension with zero elements
		x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
		x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));
		//x = torch::relu(torch::max_pool2d(conv2_drop->forward(x), 2));
		x = x.view({ -1, 320 });
		x = torch::relu(fc1->forward(x));
		x = torch::dropout(x, 0.5, is_training());
		x = torch::relu(fc2->forward(x));
		return torch::log_softmax(x, 1);
	}
	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::Conv2d conv2{ nullptr };
	//torch::nn::FeatureDropout conv2_drop{ nullptr };
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

int main() {

	// make_shared ----- owns and stores a pointer to a newly allocated object of type T
	auto net = std::make_shared<Net>();

	// loading dataset
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(torch::data::datasets::MNIST("/Users/91939/Downloads/datasets/pytorch/fashion_mnist/unzipped").map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
			torch::data::transforms::Stack<>())), 64);
	torch::optim::SGD optimizer(net->parameters(), 0.01);


	// net.train();
	for (size_t epoch = 1; epoch <= 10; ++epoch) {
		size_t batch_index = 0;
		for (auto& batch : *data_loader) {
			// Reset gradients
			optimizer.zero_grad();
			// Execute model
			torch::Tensor prediction = net->forward(batch.data);
			// Compute loss 
			torch::Tensor loss = torch::nll_loss(prediction, batch.target);
			// Compute gradients
			loss.backward();
			// update parameters
			optimizer.step();

			// printing
			if (++batch_index % 100 == 0) {
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
					<< " | Loss: " << loss.item<float>() << std::endl;
				torch::save(net, "net.pt");
			}
		}
	}
}
