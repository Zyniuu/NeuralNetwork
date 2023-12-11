#pragma once
#include "src/activations/activations.h"
#include "src/optimizers/optimizers.h"
#include "src/CSV/CSVParser.h"


namespace nn {
    enum ActivationFunction { TANH, RELU, SIGMOID, SOFTMAX };

    struct ActivationPair
    {
        ActivationFunction activation1;
        ActivationFunction activation2;

        bool operator==(const ActivationPair& other) const
        {
            return std::tie(activation1, activation2) == std::tie(other.activation1, other.activation2);
        }
    };
}


namespace std
{
	template <>
	struct hash<nn::ActivationPair>
	{
		std::size_t operator()(const nn::ActivationPair& activationPair) const
		{
			auto hash1 = std::hash<int>()(static_cast<int>(activationPair.activation1));
			auto hash2 = std::hash<int>()(static_cast<int>(activationPair.activation2));
			return hash1 ^ hash2;
		}
	};
}