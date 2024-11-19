"""
    This script contains functions for data preprocessing, such as one-hot encoding and normalization. It also contains functions for splitting the data into training and testing sets using hold-out technique.
    
    The script is divided in 3 sections:
        - One-hot encoding: Functions for one-hot encoding of the data.
        - Normalization: Functions for normalization of the data.
        - Splitting: Functions for splitting the data into training and testing sets.
"""

using Random;

""" 1.ONE HOT ENCODING """

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    """
    This function receives a feature vector and a list of classes and returns a one-hot encoding of the feature vector.
    
    Parameters:
        - feature: A vector with the class assigned to each pattern.
        - classes: A vector with the classes.
    
    Returns:
        - oneHot: A matrix with the one-hot encoding of the feature vector.
    """
    
    # Check all classes assigned to the patterns are in the list of classes
    @assert(all([in(value, classes) for value in feature]));

    # Check the number of classes
    numClasses = length(classes);
    @assert(numClasses>1)

    if (numClasses==2)
        # If we have only two classes, we can use a simple encoding in a single column
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        # If we have more than two classes, we need to use a one-hot encoding in multiple columns
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;

end


function oneHotEncoding(feature::AbstractArray{<:Any,1})
    """
    This function receives a feature vector and returns a one-hot encoding of the feature vector.
    
    Parameters:
        - feature: A vector with the class assigned to each pattern.
    
    Returns:
        - oneHot: A matrix with the one-hot encoding of the feature vector.
    """

    return oneHotEncoding(feature, unique(feature));

end

function oneHotEncoding(feature::AbstractArray{Bool,1})
    """
    This function receives a feature vector and returns the corresponding one-hot encoding column vector.
    
    Parameters:
        - feature: A vector with the class assigned to each pattern.
    
    Returns:
        - oneHot: A column vector with the one-hot encoding of the feature vector.
    """

    return reshape(feature, :, 1);

end

""" 2. NORMALIZATION """

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    """
    This function calculates the minimum and maximum values of each attribute in the dataset.
    
    Parameters:
        - dataset: A matrix with the dataset.
    
    Returns:
        - minValues: A matrix with the minimum values of each attribute.
        - maxValues: A matrix with the maximum values of each attribute.
    """
    
    return minimum(dataset, dims=1), maximum(dataset, dims=1)

end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    """
    This function calculates the mean and standard deviation of each attribute in the dataset.
    
    Parameters:
        - dataset: A matrix with the dataset.
    
    Returns:
        - avgValues: A matrix with the mean values of each attribute.
        - stdValues: A matrix with the standard deviation of each attribute.
    """
    
    return mean(dataset, dims=1), std(dataset, dims=1)

end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    """
    This function normalizes the dataset using the minimum and maximum values of each attribute, provided as input. It also eliminates any atribute that does not add information.
    
    Parameters:
        - (!)dataset: A matrix with the dataset. This matrix will be modified with the normalized values.
        - normalizationParameters: A tuple with the minimum and maximum values of each attribute.
    
    Returns:
        - dataset: A matrix with the normalized dataset.
    """
    
    # Calculate the normalization parameters
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);

    # Eliminate any atribute that does not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    
    return dataset;

end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    """
    This function normalizes the dataset using the MinMax normalization, calculating the minimum and maximum values of each attribute inside.

    Parameters:
        - (!)dataset: A matrix with the dataset. This matrix will be modified with the normalized values.
    
    Returns:
        - dataset: A matrix with the normalized dataset.
    
    """
    
    return normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));

end

function normalizeMinMax(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    """
    This function returns the normalized the dataset using the minimum and maximum values of each attribute, provided as input. It also eliminates any atribute that does not add information.

    Parameters:
        - dataset: A matrix with the dataset.
        - normalizationParameters: A tuple with the minimum and maximum values of each attribute.
    
    Returns:
        - dataset: A matrix with the normalized dataset.
    """

    return normalizeMinMax!(copy(dataset), normalizationParameters);

end

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    """
    This function returns the normalized the dataset using the MinMax normalization, calculating the minimum and maximum values of each attribute inside.
    
    Parameters:
        - dataset: A matrix with the dataset.
    
    Returns:
        - dataset: A matrix with the normalized dataset.
    """

    normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));

end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    """
    This function normalizes the dataset using the zero mean normalization, calculating the mean and standard deviation of each attribute inside. It also eliminates any atribute that does not add information.
    
    Parameters:
        - (!)dataset: A matrix with the dataset. This matrix will be modified with the normalized values.
        - normalizationParameters: A tuple with the mean and standard deviation of each attribute.
    
    Returns:
        - dataset: A matrix with the normalized dataset.
    """

    
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    
    # Remove any atribute that do not have information
    dataset[:, vec(stdValues.==0)] .= 0;
    
    return dataset; 

end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    """
    This function normalizes the dataset using the zero mean normalization, calculating the mean and standard deviation of each attribute inside.
    
    Parameters:
        - (!)dataset: A matrix with the dataset. This matrix will be modified with the normalized values.

    Returns:
        - dataset: A matrix with the normalized dataset.
    """

    return normalizeZeroMean!(dataset , calculateZeroMeanNormalizationParameters(dataset)); 

end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    """
    This function returns the normalized the dataset using the zero mean normalization, calculating the mean and standard deviation of each attribute inside. It also eliminates any atribute that does not add information.
    
    Parameters:
        - dataset: A matrix with the dataset.
        - normalizationParameters: A tuple with the mean and standard deviation of each attribute.
    
    Returns:
        - dataset: A matrix with the normalized dataset.
    """

    normalizeZeroMean!(copy(dataset), normalizationParameters);

end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    """
    This function returns the normalized the dataset using the zero mean normalization, calculating the mean and standard deviation of each attribute inside.
    
    Parameters:
        - dataset: A matrix with the dataset.
    
    Returns:
        - dataset: A matrix with the normalized dataset.  
    """

    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));

end

""" 3. SPLITTING """

function holdOut(N::Int, P::Real)
    """
    This function returns random indices for training and test sets according a percentage of the total number of patterns.
    
    Parameters:
        - N: The total number of patterns.
        - P: The percentage of patterns to be used in the test set.
    
    Returns:
        - train_indices: The indices of the training set.
        - test_indices: The indices of the test set.
    """

    # Generate a random permutation of the indices
    indices = randperm(N)
    
    # Determine the number of test samples
    test_size = Int(round(P * N))
    
    # Assign the test and training indices
    return (indices[test_size+1:end], indices[1:test_size])

end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    """
    This function returns random indices for training, validation and test sets according a percentage of the total number of patterns.
    
    Parameters:
        - N: The total number of patterns.
        - Pval: The percentage of patterns to be used in the validation set.
        - Ptest: The percentage of patterns to be used in the test set.

    Returns:
        - train_indices: The indices of the training set.
        - val_indices: The indices of the validation set.
        - test_indices: The indices of the test set.
    """

    # Split the test set
    train_val_indices, test_indices = holdOut(N, Ptest)
    
    # Split the remaining patterns into training and validation sets
    train_indices, val_indices = holdOut(size(train_val_indices, 1), Pval / (1 - Ptest))

    return (train_val_indices[train_indices], train_val_indices[val_indices], test_indices)

end

function split_data(input_data, output_data, train_ratio = 0.8)
    """
    This function splits the input and output data into training and testing sets according to the train ratio.
    
    Parameters:
        - input_data: The input data.
        - output_data: The output data.
        - train_ratio: The ratio of the training set.
    
    Returns:
        - train_input: The input data for the training set.
        - train_output: The output data for the training set.
        - test_input: The input data for the testing set.
        - test_output: The output data for the testing set.
    """

    n = size(input_data, 1)
    indices = shuffle(1:n)
    train_size = Int(floor(n * train_ratio))  # Calculate train size based on the train ratio
    train_idx = indices[1:train_size] 
    test_idx = indices[train_size+1:end]
    
    # Split the data into training and testing sets
    train_input = input_data[train_idx, :]
    train_output = output_data[train_idx]
    test_input = input_data[test_idx, :]
    test_output = output_data[test_idx]
    
    return train_input, train_output, test_input, test_output

end