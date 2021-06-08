// neural-networks.cpp : This file contains the 'main' function. Program execution begins and ends there.
// https://pytorch.org/cppdocs/frontend.html
// https://krshrimali.github.io/PyTorch-C++-API/

#include <torch\torch.h>

struct Net : torch::nn::Module {
	Net() {
		// initialization
		// register_module ----- module is registered as a submodule to another module
		fc1 = register_module("fc1", torch::nn::Linear(784, 64));
		fc2 = register_module("fc2", torch::nn::Linear(64, 32));
		fc3 = register_module("fc3", torch::nn::Linear(32, 10));
	}
	// implementation
	torch::Tensor forward(torch::Tensor x) {
		// tensor.size(0) ---- one-dimension with zero elements
		x = torch::relu(fc1->forward(x.reshape({ x.size(0), 784 })));
		x = torch::dropout(x, 0.5, is_training());
		x = torch::relu(fc2->forward(x));
		x = torch::log_softmax(fc3->forward(x), 1);
		return x;
	}
	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
};

int main() {

	// make_shared ----- owns and stores a pointer to a newly allocated object of type T
	auto net = std::make_shared<Net>();

	// loading dataset
	// multi-threaded data loader ---- bahut sare threads (smallest unit in os) ki data loader
	// samplers ---- yeilds index to access dataset
	// SequentialSampler ---- return indices in the range 0 to size-1
	// move ---- copy constructer in c++
	// map ---- combination of key value, mapped value

	//auto data_loader = torch::data::make_data_loader(
	//	torch::data::datasets::MNIST("./data").map(
	//		torch::data::transforms::Stack<>()),
	//	/*batch_size=*/64);
	
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(torch::data::datasets::MNIST("/Users/91939/Downloads/mnist_data").map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
			torch::data::transforms::Stack<>())), 64);
	torch::optim::SGD optimizer(net->parameters(), 0.01);


	// net.train();
	for (size_t epoch = 1; epoch <= 10; ++epoch) {
		size_t batch_index = 0;
		// Iterate data loader to yield batches from the dataset
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
