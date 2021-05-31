// linear regression 

#include <torch\torch.h>
#include <iostream>
#include <iomanip>

int main() {

	std::cout << "Linear Regression" << std::endl;

	// checking device
	torch::Device device = torch::kCPU;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is Available! Training on GPU." << std::endl;
		device = torch::kCUDA;
	}
	else {
		std::cout << "Training on CPU." << std::endl;
	}

	// hyperparameters
	const int64_t input_size_ = 1;
	const int64_t output_size_ = 1;
	const size_t num_epochs_ = 1000;
	const double learning_rate_ = 0.001;

	// creating sample dataset
	auto x_train_ = torch::randint(0, 10, { 15, 1 });
	auto y_train_ = torch::randint(0, 10, { 15, 1 });

	// linear regression instance
	torch::nn::Linear model(input_size_, output_size_);

	// optimizer
	torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate_));

	// Set floating point output precision
	std::cout << std::fixed << std::setprecision(4);

	std::cout << "Training..." << std::endl;

	// model training
	for (size_t epoch_ = 0; epoch_ != num_epochs_; epoch_++) {

		// forward pass
		auto output = model(x_train_);
		auto loss = torch::nn::functional::mse_loss(output, y_train_);

		// backward pass and optimize
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		if ((epoch_ + 1) % 5 == 0) {
			std::cout << "Epoch [" << (epoch_ + 1) << "/" << num_epochs_ << "], Loss: " << loss.item<double>() << std::endl;
		}

	}

	std::cout << "Training Finished!" << std::endl;

}
