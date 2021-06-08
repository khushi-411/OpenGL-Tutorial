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




/*

	Output:

	Epoch: 1 | Batch: 100 | Loss: 2.00421
	Epoch: 1 | Batch: 200 | Loss: 1.6705
	Epoch: 1 | Batch: 300 | Loss: 1.31503
	Epoch: 1 | Batch: 400 | Loss: 0.860676
	Epoch: 1 | Batch: 500 | Loss: 0.889412
	Epoch: 1 | Batch: 600 | Loss: 0.833607
	Epoch: 1 | Batch: 700 | Loss: 0.655532
	Epoch: 1 | Batch: 800 | Loss: 0.809928
	Epoch: 1 | Batch: 900 | Loss: 0.575914
	Epoch: 2 | Batch: 100 | Loss: 0.417559
	Epoch: 2 | Batch: 200 | Loss: 0.58347
	Epoch: 2 | Batch: 300 | Loss: 0.594671
	Epoch: 2 | Batch: 400 | Loss: 0.665773
	Epoch: 2 | Batch: 500 | Loss: 0.687707
	Epoch: 2 | Batch: 600 | Loss: 0.674879
	Epoch: 2 | Batch: 700 | Loss: 0.41173
	Epoch: 2 | Batch: 800 | Loss: 0.591924
	Epoch: 2 | Batch: 900 | Loss: 0.408242
	Epoch: 3 | Batch: 100 | Loss: 0.300348
	Epoch: 3 | Batch: 200 | Loss: 0.531788
	Epoch: 3 | Batch: 300 | Loss: 0.536443
	Epoch: 3 | Batch: 400 | Loss: 0.448722
	Epoch: 3 | Batch: 500 | Loss: 0.407731
	Epoch: 3 | Batch: 600 | Loss: 0.432052
	Epoch: 3 | Batch: 700 | Loss: 0.40346
	Epoch: 3 | Batch: 800 | Loss: 0.460417
	Epoch: 3 | Batch: 900 | Loss: 0.320826
	Epoch: 4 | Batch: 100 | Loss: 0.338267
	Epoch: 4 | Batch: 200 | Loss: 0.424698
	Epoch: 4 | Batch: 300 | Loss: 0.35676
	Epoch: 4 | Batch: 400 | Loss: 0.385862
	Epoch: 4 | Batch: 500 | Loss: 0.366489
	Epoch: 4 | Batch: 600 | Loss: 0.550927
	Epoch: 4 | Batch: 700 | Loss: 0.44667
	Epoch: 4 | Batch: 800 | Loss: 0.467417
	Epoch: 4 | Batch: 900 | Loss: 0.328867
	Epoch: 5 | Batch: 100 | Loss: 0.218233
	Epoch: 5 | Batch: 200 | Loss: 0.382081
	Epoch: 5 | Batch: 300 | Loss: 0.393885
	Epoch: 5 | Batch: 400 | Loss: 0.431487
	Epoch: 5 | Batch: 500 | Loss: 0.344302
	Epoch: 5 | Batch: 600 | Loss: 0.331507
	Epoch: 5 | Batch: 700 | Loss: 0.338312
	Epoch: 5 | Batch: 800 | Loss: 0.367269
	Epoch: 5 | Batch: 900 | Loss: 0.272426
	Epoch: 6 | Batch: 100 | Loss: 0.266268
	Epoch: 6 | Batch: 200 | Loss: 0.326783
	Epoch: 6 | Batch: 300 | Loss: 0.339733
	Epoch: 6 | Batch: 400 | Loss: 0.501979
	Epoch: 6 | Batch: 500 | Loss: 0.452871
	Epoch: 6 | Batch: 600 | Loss: 0.507804
	Epoch: 6 | Batch: 700 | Loss: 0.347526
	Epoch: 6 | Batch: 800 | Loss: 0.345345
	Epoch: 6 | Batch: 900 | Loss: 0.267064
	Epoch: 7 | Batch: 100 | Loss: 0.245649
	Epoch: 7 | Batch: 200 | Loss: 0.38428
	Epoch: 7 | Batch: 300 | Loss: 0.281709
	Epoch: 7 | Batch: 400 | Loss: 0.397045
	Epoch: 7 | Batch: 500 | Loss: 0.372134
	Epoch: 7 | Batch: 600 | Loss: 0.337443
	Epoch: 7 | Batch: 700 | Loss: 0.260939
	Epoch: 7 | Batch: 800 | Loss: 0.339818
	Epoch: 7 | Batch: 900 | Loss: 0.226925
	Epoch: 8 | Batch: 100 | Loss: 0.273331
	Epoch: 8 | Batch: 200 | Loss: 0.375139
	Epoch: 8 | Batch: 300 | Loss: 0.196689
	Epoch: 8 | Batch: 400 | Loss: 0.441041
	Epoch: 8 | Batch: 500 | Loss: 0.312714
	Epoch: 8 | Batch: 600 | Loss: 0.274101
	Epoch: 8 | Batch: 700 | Loss: 0.297219
	Epoch: 8 | Batch: 800 | Loss: 0.324144
	Epoch: 8 | Batch: 900 | Loss: 0.185917
	Epoch: 9 | Batch: 100 | Loss: 0.242066
	Epoch: 9 | Batch: 200 | Loss: 0.315567
	Epoch: 9 | Batch: 300 | Loss: 0.218419
	Epoch: 9 | Batch: 400 | Loss: 0.34342
	Epoch: 9 | Batch: 500 | Loss: 0.31547
	Epoch: 9 | Batch: 600 | Loss: 0.417608
	Epoch: 9 | Batch: 700 | Loss: 0.206981
	Epoch: 9 | Batch: 800 | Loss: 0.362743
	Epoch: 9 | Batch: 900 | Loss: 0.241284
	Epoch: 10 | Batch: 100 | Loss: 0.221403
	Epoch: 10 | Batch: 200 | Loss: 0.387969
	Epoch: 10 | Batch: 300 | Loss: 0.189302
	Epoch: 10 | Batch: 400 | Loss: 0.4035
	Epoch: 10 | Batch: 500 | Loss: 0.292447
	Epoch: 10 | Batch: 600 | Loss: 0.317219
	Epoch: 10 | Batch: 700 | Loss: 0.202563
	Epoch: 10 | Batch: 800 | Loss: 0.345983
	Epoch: 10 | Batch: 900 | Loss: 0.188844

	C:\Users\91939\source\repos\neural-networks\x64\Debug\neural-networks.exe (process 22300) exited with code 0.
	Press any key to close this window . . .
*/

