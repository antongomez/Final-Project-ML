"""
    This script contains the functions to build and train a variety of classification models.

    The script is divided in 4 sections:
    1. ANN: Contains the functions to build and train an Artificial Neural Network.
    2. EARLY STOPPING: Contains the functions to train an ANN with early stopping.
    3. CROSS VALIDATION TRAINING: Contains the functions to train an ANN and other models with k-fold cross-validation.
    4. ENSEMBLE: Contains the functions to train ensemble models using k-fold cross-validation.
"""


using Flux;
using Flux.Losses;
using ScikitLearn;

include("metrics.jl");
include("preprocessing.jl");

# Base models
@sk_import svm:SVC
@sk_import tree:DecisionTreeClassifier
@sk_import neighbors:KNeighborsClassifier
@sk_import linear_model:LogisticRegression
@sk_import naive_bayes:GaussianNB
@sk_import neural_network:MLPClassifier

# Ensemble models
@sk_import ensemble:(AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier, VotingClassifier, RandomForestClassifier)

# PCA for dimensionality reduction
@sk_import decomposition:PCA


""" 1. ANN """

function buildClassANN(numInputs::Int, 
    topology::AbstractArray{<:Int,1}, 
    numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    """
    This function builds a neural_network with the specfied parameters.
    
    Parameters:
        - numInputs: Number of inputs of the network.
        - topology: Array with the number of neurons in each hidden layer.
        - numOutputs: Number of outputs of the network.
        - transferFunctions: Array with the transfer functions of each layer.
    Returns:
        - ann: The neural network.
    """
    
    @assert length(transferFunctions) == length(topology)

    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;

end;

function calculate_and_log_loss(model,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2},
    label::String,
    loss_function::Function,
    showText::Bool;
    epoch::Int=-1
)
    """
    This function calculates the loss of the model and logs it to the console.
    
    Parameters:
        - model: The model to calculate the loss.
        - inputs: The inputs to calculate the loss.
        - targets: The targets to calculate the loss.
        - label: The label to show in the console.
        - loss_function: The loss function to calculate the loss.
        - showText: A boolean to show the loss in the console.
        - epoch: The epoch number to show in the console.
    
    Returns:
        - loss_value: The value of the loss.
    """
    loss_value = loss_function(model, inputs', targets')
    if showText
        if epoch > -1
            println("Epoch $epoch:")
        end
        println("\t$label Loss: ", loss_value)
    end
    return loss_value;
end

function trainClassANN(topology::AbstractArray{<:Int,1},      
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,
    showText::Bool=false) 
    """
    This function trains a neural network with the specified parameters only with the training dataset, receiving the inputs and targets as both real and boolean matrixes, respectively.
    
    Parameters:
        - topology: Array with the number of neurons in each hidden layer.
        - dataset: Tuple with the inputs and targets of the dataset, both as real and boolean matrixes, respectively.
        - transferFunctions: Array with the transfer functions of each layer.
        - maxEpochs: The maximum number of epochs to train the network.
        - minLoss: The minimum loss to stop the training.
        - learningRate: The learning rate of the network.
        - showText: A boolean to show the loss in the console.

    Returns:
        - ann: The trained neural network.
        - trainingLosses: The losses of the training in each epoch.
    """

    (inputs, targets) = dataset;

    # This function assumes that each sample is in a row
    # we are going to check the numeber of samples to have same inputs and targets
    @assert(size(inputs,1)==size(targets,1));

    ann = buildClassANN(size(inputs,2), topology, size(targets,2), transferFunctions = transferFunctions);

    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    trainingLosses = Float32[];
    numEpoch = 0;

    # Calcualte the initial loss
    trainingLoss = calculate_and_log_loss(ann, inputs, targets, "Training", loss, showText, epoch = numEpoch);
    push!(trainingLosses, trainingLoss);

    opt_state = Flux.setup(Adam(learningRate), ann);

    # Start the training until it reaches one of the stop critteria
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        # For each epoch, we habve to train and consequently traspose the pattern to have then in columns
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);

        numEpoch += 1;
        trainingLoss = calculate_and_log_loss(ann, inputs, targets, "Training", loss, showText, epoch = numEpoch);
        push!(trainingLosses, trainingLoss);

    end;
    return (ann, trainingLosses);

end;

function trainClassANN(topology::AbstractArray{<:Int,1},      
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};      
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),      
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,
    showText::Bool=false) 
    """
    This function trains a neural network with the specified parameters only with the training dataset, receiving the inputs and targets as real matrix and boolean vector, respectively.

    Parameters:
        - topology: Array with the number of neurons in each hidden layer.
        - inputs: A real matrix of the dataset.
        - targets: A boolean array targets of the dataset.
        - transferFunctions: Array with the transfer functions of each layer.
        - maxEpochs: The maximum number of epochs to train the network.
        - minLoss: The minimum loss to stop the training.
        - learningRate: The learning rate of the network.
        - showText: A boolean to show the loss in the console.
    
    Returns:
        - ann: The trained neural network.
        - trainingLosses: The losses of the training in each epoch.
    """
    
    return trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, showText=showText);

end;

""" 2. EARLY STOPPING """

function trainClassANN(topology::AbstractArray{<:Int,1},  
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,  
    maxEpochsVal::Int=20, 
    showText::Bool=false)
    """
    This function trains a neural network with the specified parameters with the training, validation, and test datasets, receiving the inputs and targets as both real and boolean matrixes, respectively, and performing early stopping with the validation dataset.

    Parameters:
        - topology: Array with the number of neurons in each hidden layer.
        - trainingDataset: Tuple with the inputs and targets of the training dataset, both as real and boolean matrixes, respectively.
        - validationDataset: Tuple with the inputs and targets of the validation dataset, both as real and boolean matrixes, respectively.
        - testDataset: Tuple with the inputs and targets of the test dataset, both as real and boolean matrixes, respectively.
        - transferFunctions: Array with the transfer functions of each layer.
        - maxEpochs: The maximum number of epochs to train the network.
        - minLoss: The minimum loss to stop the training.
        - learningRate: The learning rate of the network.
        - maxEpochsVal: The maximum number of epochs without improvement in the validation loss to stop the training.
        - showText: A boolean to show the loss in the console.
    
    Returns:
        - finalModel: The trained neural network.
        - trainingLosses: The losses of the training in each epoch.
        - validationLosses: The losses of the validation in each epoch.
        - testLosses: The losses of the test in each epoch.
        - bestEpoch: The epoch with the best validation loss.
    """

    (inputs, targets) = trainingDataset;
    # Check that the number of samples matches in inputs and targets
    @assert(size(inputs,1) == size(targets,1));

    # Check if validation and test sets have been provided and if they have the correct dimensions
    val_set = false
    test_set = false
    if !isempty(validationDataset[1])
        (val_inputs, val_targets) = validationDataset
        val_set = true
        @assert(size(val_inputs, 1) == size(val_targets, 1))
        @assert(size(val_inputs, 2) == size(inputs, 2))
        @assert(size(val_targets, 2) == size(targets, 2))
    end
    if !isempty(testDataset[1])
        (test_inputs, test_targets) = testDataset
        test_set = true
        @assert(size(test_inputs, 1) == size(test_targets, 1))
        @assert(size(test_inputs, 2) == size(inputs, 2))
        @assert(size(test_targets, 2) == size(targets, 2))
    end

    ann = buildClassANN(size(inputs,2), topology, size(targets,2), transferFunctions = transferFunctions);
    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y);
    opt_state = Flux.setup(Adam(learningRate), ann);

    # Initialize training, validation, and test loss history
    trainingLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];

    # Initialize counters and variables for early stopping
    numEpoch = 0;
    bestValLoss = Inf;
    bestModel = deepcopy(ann);
    bestEpoch = 0;
    epochsWithoutImprovement = 0;

    # Show initial feedback
    push!(trainingLosses, calculate_and_log_loss(ann, inputs, targets, "Training", loss, showText, epoch = numEpoch))
    if val_set
        push!(validationLosses, calculate_and_log_loss(ann, val_inputs, val_targets, "Validation", loss, showText, epoch = numEpoch))
    end
    if test_set
        push!(testLosses, calculate_and_log_loss(ann, test_inputs, test_targets, "Test", loss, showText, epoch = numEpoch))
    end

    # Training loop
    while (numEpoch < maxEpochs) && (trainingLosses[end] > minLoss)

        Flux.train!(loss, ann, [(inputs', targets')], opt_state);
        numEpoch += 1;
        push!(trainingLosses, calculate_and_log_loss(ann, inputs, targets, "Training", loss, showText, epoch = numEpoch));

        if val_set
            push!(validationLosses, calculate_and_log_loss(ann, val_inputs, val_targets, "Validation", loss, showText, epoch = numEpoch));

            # Early stopping: Check if validation loss improves
            if validationLosses[end] < bestValLoss
                bestValLoss = validationLosses[end];
                bestModel = deepcopy(ann);
                bestEpoch = numEpoch;
                epochsWithoutImprovement = 0;
            else
                epochsWithoutImprovement += 1;
                if epochsWithoutImprovement >= maxEpochsVal
                    if showText
                        println("Early stopping triggered at epoch $numEpoch. Best validation loss: $bestValLoss (at epoch $(numEpoch - maxEpochsVal))")
                    end
                    break
                end
            end

        end

        if test_set
            push!(testLosses, calculate_and_log_loss(ann, test_inputs, test_targets, "Test", loss, showText, epoch = numEpoch));
        end
        
    end

    if val_set
        finalModel = bestModel;
    else
        finalModel = ann;
    end

    return (finalModel, trainingLosses, validationLosses, testLosses, bestEpoch);

end

function trainClassANN(topology::AbstractArray{<:Int,1},  
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,  
    maxEpochsVal::Int=20, 
    showText::Bool=false)
    """
    This function trains a neural network with the specified parameters with the training, validation, and test datasets, receiving the inputs as a real matrix and the targets as a boolean vector, and performing early stopping with the validation dataset.

    Parameters:
        - topology: Array with the number of neurons in each hidden layer.
        - trainingDataset: Tuple with the inputs and targets of the training dataset, the inputs as a real matrix and the targets as a boolean vector.
        - validationDataset: Tuple with the inputs and targets of the validation dataset, the inputs as a real matrix and the targets as a boolean vector.
        - testDataset: Tuple with the inputs and targets of the test dataset, the inputs as a real matrix and the targets as a boolean vector.
        - transferFunctions: Array with the transfer functions of each layer.
        - maxEpochs: The maximum number of epochs to train the network.
        - minLoss: The minimum loss to stop the training.
        - learningRate: The learning rate of the network.
        - maxEpochsVal: The maximum number of epochs without improvement in the validation loss to stop the training.
        - showText: A boolean to show the loss in the console.
    
    Returns:
        - finalModel: The trained neural network.
        - trainingLosses: The losses of the training in each epoch.
        - validationLosses: The losses of the validation in each epoch.
        - testLosses: The losses of the test in each epoch.
        - bestEpoch: The epoch with the best validation loss.
    """


    # Reshape the targets from vectors to arrays (length(targets), 1)
    reshapedTrainingDataset = (trainingDataset[1], reshape(trainingDataset[2], length(trainingDataset[2]), 1))
    reshapedValidationDataset = (validationDataset[1], reshape(validationDataset[2], length(validationDataset[2]), 1))
    reshapedTestDataset = (testDataset[1], reshape(testDataset[2], length(testDataset[2]), 1))

    # Reuse the main trainClassANN function with reshaped targets
    trainClassANN(topology, reshapedTrainingDataset; 
        validationDataset=reshapedValidationDataset, 
        testDataset=reshapedTestDataset, 
        transferFunctions=transferFunctions, 
        maxEpochs=maxEpochs, 
        minLoss=minLoss, 
        learningRate=learningRate, 
        maxEpochsVal=maxEpochsVal, 
        showText=showText);

end

""" 3. CROSS VALIDATION TRAINING """

function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
    kFoldIndices::Array{Int64,1}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01, 
    repetitionsTraining::Int=1, 
    validationRatio::Real=0.0, 
    maxEpochsVal::Int=20, 
    metricsToSave::AbstractArray{<:String, 1}=["accuracy"], 
    showText = false, 
    showTextEpoch = false,
    normalizationType::Symbol=:zeroMean)
    """
    This function trains a neural network with the specified parameters with the training dataset using k-fold cross-validation, receiving the inputs and targets as both real and boolean matrixes, respectively.
    
    Parameters:
        - topology: Array with the number of neurons in each hidden layer.
        - trainingDataset: Tuple with the inputs and targets of the training dataset, both as real and boolean matrixes, respectively.
        - kFoldIndices: Array with the indices of the k-fold cross-validation.
        - transferFunctions: Array with the transfer functions of each layer.
        - maxEpochs: The maximum number of epochs to train the network.
        - minLoss: The minimum loss to stop the training.
        - learningRate: The learning rate of the network.
        - repetitionsTraining: The number of repetitions to train the network.
        - validationRatio: The ratio of the validation dataset.
        - maxEpochsVal: The maximum number of epochs without improvement in the validation loss to stop the training.
        - metricsToSave: Array with the metrics to save.
        - showText: A boolean to show the loss in the console.
        - showTextEpoch: A boolean to show the loss in the console in each epoch.
        - normalizationType: The type of normalization to apply to the data.
    
    Returns:
        - mean_results: The mean of each metric for each fold.
        - std_results: The standard deviation of each metric for each fold.
    """
    # Get the number of folds
    n_folds = maximum(kFoldIndices)

    # Create a dictionary to store each metric's evaluations
    results_fold = Dict{String, AbstractArray{Float64, 1}}()

    # Initialize each selected metric in the dictionary
    for metric in metricsToSave
        results_fold[metric] = zeros(Float64, n_folds)
    end

    # For each fold
    for i in 1:n_folds
        if showText
            println("Fold ", i, ":")
        end
        # Here we will store the results for each metric on each repetition
        results_iterations = Dict{String, AbstractArray{Float64, 1}}()
        for metric in metricsToSave
            results_iterations[metric] = zeros(Float64, repetitionsTraining)
        end

        # Get the training and test datasets for the current fold
        trainingDatasetFold = (trainingDataset[1][kFoldIndices .!= i, :], trainingDataset[2][kFoldIndices .!= i, :])
        testDatasetFold = (trainingDataset[1][kFoldIndices .== i, :], trainingDataset[2][kFoldIndices .== i, :])

        # If validationRatio is greater than 0, split the training dataset into training and validation
        if validationRatio > 0

            train_indices, val_indices = holdOut(size(trainingDatasetFold[1], 1), validationRatio)
            validationDatasetFold = (trainingDatasetFold[1][val_indices, :], trainingDatasetFold[2][val_indices, :])
            trainingDatasetFold = (trainingDatasetFold[1][train_indices, :], trainingDatasetFold[2][train_indices, :])

            # Normalize the data
            normalizationParameters = calculateNormalizationParameters(trainingDatasetFold[1], normalizationType);
            performNormalization!(trainingDatasetFold[1], normalizationParameters, normalizationType);
            performNormalization!(validationDatasetFold[1], normalizationParameters, normalizationType);
            performNormalization!(testDatasetFold[1], normalizationParameters, normalizationType);

            # Reduce dimension with pca
            pca = PCA(n_components=0.95)
            fit!(pca, trainingDatasetFold[1])
            pca.transform(trainingDatasetFold[1])
            pca.transform(validationDatasetFold[1])

            for j in 1:repetitionsTraining
                (model, trainingLosses, validationLosses, testLosses, best_epoch) = trainClassANN(topology, trainingDatasetFold, validationDataset=validationDatasetFold, testDataset=testDatasetFold, transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=showTextEpoch)
                outputs = model(testDatasetFold[1]')'
                matrix, metrics = confusionMatrix(outputs, testDatasetFold[2])
                for metric in metricsToSave
                    results_iterations[metric][j] = metrics[metric]
                end
                if showText
                    println("\tRepetition ", j, ": ")
                    for metric in metricsToSave
                        println("\t\t", metric, ": ", results_iterations[metric][j])
                    end
                end
            end
        
        # Otherwise, train the model without validation
        else
            # Normalize the data
            normalizationParameters = calculateNormalizationParameters(trainingDatasetFold[1], normalizationType);
            performNormalization!(trainingDatasetFold[1], normalizationParameters, normalizationType);
            performNormalization!(testDatasetFold[1], normalizationParameters, normalizationType);

            # Reduce dimension with pca
            pca = PCA(n_components=0.95)
            fit!(pca, trainingDatasetFold[1])
            pca.transform(trainingDatasetFold[1])
            pca.transform(testDatasetFold[1])
            
            for j in 1:repetitionsTraining
                (model, trainingLosses, validationLosses, testLosses, best_epoch) = trainClassANN(topology, trainingDatasetFold, testDataset=testDatasetFold, transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=showTextEpoch)
                outputs = model(testDatasetFold[1]')'
                matrix, metrics = confusionMatrix(outputs, testDatasetFold[2])
                for metric in metricsToSave
                    results_iterations[metric][j] = metrics[metric]
                end
                if showText
                    println("\tRepetition ", j, ": ")
                    for metric in metricsToSave
                        println("\t\t", metric, ": ", results_iterations[metric][j])
                    end
                end
            end
        end

        # Store the results for the fold
        for metric in metricsToSave
            results_fold[metric][i] = mean(results_iterations[metric])
        end
        if showText
            println("\tMean results for fold ", i, ":")
            for metric in metricsToSave
                println("\t\t$metric: ", results_fold[metric][i])
            end
        end

    end

    # Return the mean of each metric for each fold
    mean_results = Dict{String, Float64}()
    std_results = Dict{String, Float64}()
    for metric in metricsToSave
        mean_results[metric] = mean(results_fold[metric])
        std_results[metric] = std(results_fold[metric])
        println("Mean $metric: ", mean_results[metric], " ± ", std_results[metric])
    end

    return mean_results, std_results
    
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    kFoldIndices::	Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01,
    repetitionsTraining::Int=1, 
    validationRatio::Real=0.0, 
    maxEpochsVal::Int=20, 
    metricsToSave::AbstractArray{<:String, 1}=["accuracy"], 
    showText = false,
    showTextEpoch = false,
    normalizationType::Symbol=:zeroMean)
    """
    This function trains a neural network with the specified parameters with the training dataset using k-fold cross-validation, receiving the inputs as a real matrix and the targets as a boolean vector.

    Parameters:
        - topology: Array with the number of neurons in each hidden layer.
        - trainingDataset: Tuple with the inputs and targets of the training dataset, the inputs as a real matrix and the targets as a boolean vector.
        - kFoldIndices: Array with the indices of the k-fold cross-validation.
        - transferFunctions: Array with the transfer functions of each layer.
        - maxEpochs: The maximum number of epochs to train the network.
        - minLoss: The minimum loss to stop the training.
        - learningRate: The learning rate of the network.
        - repetitionsTraining: The number of repetitions to train the network.
        - validationRatio: The ratio of the validation dataset.
        - maxEpochsVal: The maximum number of epochs without improvement in the validation loss to stop the training.
        - metricsToSave: Array with the metrics to save.
        - showText: A boolean to show the loss in the console.
        - normalizationType: The type of normalization to apply to the data.
    
    Returns:
        - mean_results: The mean of each metric for each fold.
        - std_results: The standard deviation of each metric for each fold.
    """

    (trainingInputs, trainingTargets) = trainingDataset

    # Reshape the trainingTargets to work as a 2D matrix (length, 1) if it’s a 1D vector
    reshapedTargets = reshape(trainingTargets, length(trainingTargets), 1)

    return trainclassANN(topology, (trainingInputs, reshapedTargets), kFoldIndices;
    transferFunctions = transferFunctions,
    maxEpochs = maxEpochs, 
    minLoss = minLoss, 
    learningRate = learningRate, 
    repetitionsTraining = repetitionsTraining, 
    validationRatio = validationRatio, 
    maxEpochsVal = maxEpochsVal, 
    metricsToSave = metricsToSave, 
    showText = showText,
    showTextEpoch = showTextEpoch,
    normalizationType = normalizationType)
end

function modelCrossValidation(
    modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real, 2},
    targets::AbstractArray{<:Any, 1},
    crossValidationIndices::Array{Int64, 1};
    metricsToSave::AbstractArray{<:Union{String, Symbol}, 1} = [:accuracy],
    normalizationType::Symbol = :zeroMean,
    applyPCA::Bool = false,
    pcaThreshold::Float64 = 0.95)
    """
    This function performs cross-validation for a given model with the specified hyperparameters, inputs, and targets.
    
    Parameters:
        - modelType: The type of the model to train.
        - modelHyperparameters: The hyperparameters of the model.
        - inputs: The inputs of the dataset.
        - targets: The targets of the dataset.
        - crossValidationIndices: The indices of the cross-validation.
        - metricsToSave: The metrics to save.
        - showTextEpoch: A boolean to show the loss in the console in each epoch.
        - normalizationType: The type of normalization to apply to the data.
    
    Returns:
        - mean_results: The mean of each metric for each fold.
        - std_results: The standard deviation of each metric for each fold.
    """
  
    # Get the number of folds
    n_folds = maximum(crossValidationIndices)

    # Get the number of classes
    numClasses = length(unique(targets))
  
    # Create a dictionary to store each metric's evaluations
    results_fold = Dict{Symbol, AbstractArray{Float64,1}}()
    
    # Create a vector in which each element is a dictionary where the key is the metric and the value is an array with the metric values for each fold
    class_fold_results = [
        Dict(metric => rand(Float64, n_folds) for metric in metricsToSave) for _ in 1:numClasses
    ]

    # Initialize each selected metric in the dictionary
    for metric in metricsToSave
        results_fold[metric] = zeros(Float64, n_folds)
    end
  
    # Convert the targets to oneHot encoding for the ANN
    if modelType == :ANN
      targets = oneHotEncoding(targets)
    end
  
    # For each fold
    for i in 1:n_folds
  
        if modelType == :ANN
            # Separate training and validation data for this fold
            trainIdx = findall(crossValidationIndices .!= i)
            testIdx = findall(crossValidationIndices .== i)

            # Get the training and test datasets
            trainingDatasetFold = (inputs[trainIdx, :], targets[trainIdx, :])
            testDatasetFold = (inputs[testIdx, :], targets[testIdx, :])

            # Check mandatory hyperparameters
            topology = modelHyperparameters["topology"]
  
            # Check the optional hyperparameters
            repetitionsTraining = get(modelHyperparameters, "repetitionsTraining", 1)
            validationRatio = get(modelHyperparameters, "validationRatio", 0)
            maxEpochs = get(modelHyperparameters, "maxEpochs", nothing)
            minLoss = get(modelHyperparameters, "minLoss", nothing)
            learningRate = get(modelHyperparameters, "learningRate", nothing)
            maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", nothing)
            transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology)))
            
            # Here we will store the results for each metric on each repetition
            results_iterations = Dict{Symbol, AbstractArray{Float64,1}}()

            # Initialize each selected metric in the dictionary
            for metric in metricsToSave
                results_iterations[metric] = zeros(Float64, repetitionsTraining)
            end

            # Create a vector in which each element is a dictionary where the key is the metric and the value is an array with the metric values for each repetition
            class_iterations_results = [
                Dict(metric => rand(Float64, repetitionsTraining) for metric in metricsToSave) for _ in 1:numClasses
            ]
  
            if validationRatio > 0
                train_indices, val_indices = holdOut(size(trainingDatasetFold[1], 1), validationRatio)
                validationDatasetFold = (trainingDatasetFold[1][val_indices, :], trainingDatasetFold[2][val_indices, :])
                trainingDatasetFold = (trainingDatasetFold[1][train_indices, :], trainingDatasetFold[2][train_indices, :])

                # Normalize the data
                normalizationParameters = calculateNormalizationParameters(trainingDatasetFold[1], normalizationType)
                performNormalization!(trainingDatasetFold[1], normalizationParameters, normalizationType)
                performNormalization!(validationDatasetFold[1], normalizationParameters, normalizationType)
                performNormalization!(testDatasetFold[1], normalizationParameters, normalizationType)

                # Reduce dimension with pca
                if applyPCA
                    pca = PCA(n_components=pcaThreshold)
                    fit!(pca, trainingDatasetFold[1])
                    train_inputs = pca.transform(trainingDatasetFold[1])
                    validation_inputs = pca.transform(validationDatasetFold[1])
                    test_inputs = pca.transform(testDatasetFold[1])
                    trainingDatasetFold = (train_inputs, trainingDatasetFold[2])
                    validationDatasetFold = (validation_inputs, validationDatasetFold[2])
                    testDatasetFold = (test_inputs, testDatasetFold[2])
                end

            else
                # Normalize the data
                normalizationParameters = calculateNormalizationParameters(trainingDatasetFold[1], normalizationType)
                performNormalization!(trainingDatasetFold[1], normalizationParameters, normalizationType)
                performNormalization!(testDatasetFold[1], normalizationParameters, normalizationType)

                # Reduce dimension with pca
                if applyPCA
                    pca = PCA(n_components=pcaThreshold)
                    fit!(pca, trainingDatasetFold[1])
                    train_inputs = pca.transform(trainingDatasetFold[1])
                    test_inputs = pca.transform(testDatasetFold[1])
                    trainingDatasetFold = (train_inputs, trainingDatasetFold[2])
                    testDatasetFold = (test_inputs, testDatasetFold[2])
                end
            end

            for j in 1:repetitionsTraining
                if validationRatio > 0
                    (model, trainingLosses, validationLosses, testLosses, best_epoch) = trainClassANN(topology, trainingDatasetFold, validationDataset=validationDatasetFold, testDataset=testDatasetFold, transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
                else
                    (model, trainingLosses, validationLosses, testLosses, best_epoch) = trainClassANN(topology, trainingDatasetFold, testDataset=testDatasetFold, transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
                end
                outputs = model(testDatasetFold[1]')'
                metrics, classes_results = confusionMatrix(outputs, testDatasetFold[2])
                
                for class in 1:numClasses
                    for metric in metricsToSave
                        class_iterations_results[class][metric][j] = classes_results[class][metric]
                    end
                end
                
                for metric in metricsToSave
                    results_iterations[metric][j] = metrics[metric]
                end
            end
  
        else
            # Get the training and test datasets
            trainingDatasetFold = (inputs[crossValidationIndices.!=i, :], targets[crossValidationIndices.!=i])
            testDatasetFold = (inputs[crossValidationIndices.==i, :], targets[crossValidationIndices.==i])

            # Normalize the data
            normalizationParameters = calculateNormalizationParameters(trainingDatasetFold[1], normalizationType)
            performNormalization!(trainingDatasetFold[1], normalizationParameters, normalizationType)
            performNormalization!(testDatasetFold[1], normalizationParameters, normalizationType)

            # Reduce dimension with pca
            if applyPCA
                pca = PCA(n_components=pcaThreshold)
                fit!(pca, trainingDatasetFold[1])
                train_inputs = pca.transform(trainingDatasetFold[1])
                test_inputs = pca.transform(testDatasetFold[1])
                trainingDatasetFold = (train_inputs, trainingDatasetFold[2])
                testDatasetFold = (test_inputs, testDatasetFold[2])
            end

            # Create the SVC model with the specified hyperparameters
            if modelType == :SVC
                model = SVC(; modelHyperparameters...)
            elseif modelType == :DT
                model = DecisionTreeClassifier(; modelHyperparameters...)
            elseif modelType == :KNN
                model = KNeighborsClassifier(; modelHyperparameters...)
            elseif modelType == :scikit_ANN
                model = MLPClassifier(; modelHyperparameters...)
            elseif modelType == :LR
                model = LogisticRegression(; modelHyperparameters...)
            elseif modelType == :NB
                model = GaussianNB(; modelHyperparameters...)
            else
                error("Model type: $modelType not supported.")
            end
            fit!(model, trainingDatasetFold[1], trainingDatasetFold[2])
  
            # Make predictions
            outputs = predict(model, testDatasetFold[1])
  
            # Calculate metrics
            metrics, classes_results = confusionMatrix(outputs, testDatasetFold[2])
        end

        println("Mean results for fold ", i, ":")
        for metric in metricsToSave
            if modelType == :ANN
                results_fold[metric][i] = mean(results_iterations[metric])
                println("\t$metric: ", results_fold[metric][i])
                for class in 1:numClasses
                    println("\t\tClass ", class, ": ", mean(class_iterations_results[class][metric]))
                    class_fold_results[class][metric][i] = mean(class_iterations_results[class][metric])
                end
            else
                results_fold[metric][i] = metrics[metric]
                println("\t$metric: ", results_fold[metric][i])
                for class in 1:numClasses
                    println("\t\tClass ", class, ": ", classes_results[class][metric])
                    class_fold_results[class][metric][i] = classes_results[class][metric]
                end
            end
        end
    end
  
    # Return the mean of each metric for each fold
    mean_results = Dict{Symbol, Float64}()
    std_results = Dict{Symbol, Float64}()

    for metric in metricsToSave
        mean_results[metric] = mean(results_fold[metric])
        std_results[metric] = std(results_fold[metric])
        println("Mean $metric: ", mean_results[metric], " ± ", std_results[metric])
        for class in 1:numClasses
            mean_class = mean(class_fold_results[class][metric])
            std_class = std(class_fold_results[class][metric])
            println("\tClass ", class, ": ", mean_class, " ± ", std_class)
        end   
    end
  
    return results_fold, class_fold_results
  
end

""" 4. ENSEMBLE """

function get_base_model(model_symbol, modelHyperParameters, index)
    """
    This function returns a base model with the specified hyperparameters.

    Parameters:
        - model_symbol: The symbol of the model to train.
        - modelHyperParameters: The hyperparameters of the model.
        - index: The index of the model.
    
    Returns:
        - name: The name of the model.
        - model: The model with the specified hyperparameters.
    """
    
    if model_symbol == :DT
        name = "DT$index"
        return name, DecisionTreeClassifier(; modelHyperParameters...)
    elseif model_symbol == :SVC
        name = "SVC$index"
        return name, SVC(; modelHyperParameters...)
    elseif model_symbol == :KNN
        name = "KNN$index"
        return name, KNeighborsClassifier(; modelHyperParameters...)
    elseif model_symbol == :LR
        name = "LR$index"
        return name, LogisticRegression(; modelHyperParameters...)
    elseif model_symbol == :NB
        name = "NB$index"
        return name, GaussianNB(; modelHyperParameters...)
    elseif model_symbol == :ANN
        name = "ANN$index"
        println("Get base model", typeof(modelHyperParameters))
        modelHyperParameters[:validation_fraction] = get(modelHyperParameters, :validation_fraction, 0)
        if modelHyperParameters[:validation_fraction] == 0
            modelHyperParameters[:early_stopping] = true
        end
        return name, MLPClassifier(; modelHyperParameters...)
    else
        error("Base model not allowed. Choose one of the following: [:DT, :SVC, :KNN, :LR, :NB, :ANN]")
    end
end

function update_metrics!(predictions, targets, metricsToSave, results_fold, i, showText)
    """
    This function updates the metrics of the current fold.
    
    Parameters:
        - predictions: The predictions of the model.
        - targets: The targets of the dataset.
        - metricsToSave: The metrics to save.
        - (!)results_fold: The results of the current fold. This is a mutable variable.
        - i: The index of the current fold.
        - showText: A boolean to show the loss in the console.
    
    Returns:
        - results_fold: The results adding the one of the current fold.
    """
    # if isa(predictions, AbstractVector)
    #     predictions = reshape(predictions, :, 1)
    # end
    # println(predictions)
    # println(targets)
    metrics, _ = confusionMatrix(predictions, targets)
    for metric in metricsToSave
        results_fold[metric][i] = metrics[metric]
        if showText
            println("\t", metric, ": ", results_fold[metric][i])
        end
    end
end

function get_ensemble_model(ensemble_symbol, ensembleHyperParameters, estimators, modelsHyperParameters)
    """
    This function returns an ensemble model with the specified hyperparameters.
    
    Parameters:
        - ensemble_symbol: The symbol of the ensemble model to train.
        - ensembleHyperParameters: The hyperparameters of the ensemble model.
        - estimators: The estimators of the ensemble model.
        - modelsHyperParameters: The hyperparameters of the base models.
    
    Returns:
        - ensemble: The ensemble model with the specified hyperparameters.
    """
    if ensemble_symbol == :Voting
        base_models = []
        for (i, (modelType, modelHyperParameters)) in enumerate(zip(estimators, modelsHyperParameters))
            modelName, model = get_base_model(modelType, modelHyperParameters, i)
            push!(base_models, (modelName, model))
        end
        ensembleHyperParameters[:estimators] = base_models
        return VotingClassifier(; ensembleHyperParameters...)
    elseif ensemble_symbol == :Stacking
        base_models = []
        for (i, (modelType, modelHyperParameters)) in enumerate(zip(estimators, modelsHyperParameters))
            modelName, model = get_base_model(modelType, modelHyperParameters, i)
            push!(base_models, (modelName, model))
        end
        ensembleHyperParameters[:estimators] = base_models
        return StackingClassifier(; ensembleHyperParameters...)
    elseif ensemble_symbol == :Bagging
        base_estimator = get_base_model(estimators[1], modelsHyperParameters[1], 1)[2]
        ensembleHyperParameters[:base_estimator] = base_estimator
        return BaggingClassifier(; ensembleHyperParameters...)
    elseif ensemble_symbol == :AdaBoost
        base_estimator = get_base_model(estimators[1], modelsHyperParameters[1], 1)[2]
        ensembleHyperParameters[:base_estimator] = base_estimator
        return AdaBoostClassifier(; ensembleHyperParameters...)
    elseif ensemble_symbol == :GradientBoosting
        return GradientBoostingClassifier(; ensembleHyperParameters...)
    else
        error("Ensemble model not allowed. Choose one of the following: [:Voting, :Stacking, :Bagging, :AdaBoost, :GradientBoosting]")
    end
end

function trainClassEnsemble(
    estimators::AbstractArray{Symbol, 1},
    modelsHyperParameters::Vector{Dict},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any, 1}},
    kFoldIndices::Array{Int64, 1};
    ensembleType::Symbol=:Voting,
    ensembleHyperParameters::Dict=Dict(),
    metricsToSave::AbstractArray{Symbol, 1}=["accuracy"],
    showText::Bool=false,
    normalizationType::Symbol=:zeroMean,
    applyPCA::Bool=false,
    pcaThreshold::Float64=0.95
)
    """
    This function trains an ensemble model with the specified parameters with the training dataset using k-fold cross-validation, receiving the inputs and targets as both real and boolean matrices, respectively.
    
    Parameters:
        - estimators: The estimators of the ensemble model.
        - modelsHyperParameters: The hyperparameters of the base models.
        - trainingDataset: Tuple with the inputs and targets of the training dataset, both as real and boolean matrices, respectively.
        - kFoldIndices: Array with the indices of the k-fold cross-validation.
        - ensembleType: The type of the ensemble model.
        - ensembleHyperParameters: The hyperparameters of the ensemble model.
        - metricsToSave: The metrics to save.
        - showText: A boolean to show the loss in the console.
        - normalizationType: The type of normalization to apply to the data.
        - applyPCA: Whether to apply PCA for dimensionality reduction.
        - pcaThreshold: The PCA variance threshold for dimensionality reduction.
    
    Returns:
        - results_fold: A dictionary with the results of each fold for each metric.
    """
    
    # Check if the ensemble type is allowed
    ensembles_allowed = [:Voting, :Stacking, :Bagging, :AdaBoost, :GradientBoosting]
    if !(ensembleType in ensembles_allowed)
        error("Ensemble type not allowed. Please use one of the following: $ensembles_allowed")
    end

    # Split the training dataset into inputs and targets
    inputs, targets = trainingDataset

    # Get the number of folds
    n_folds = maximum(kFoldIndices)

    # Initialize dictionary to store results of each metric
    results_fold = Dict{Symbol,AbstractArray{Float64,1}}()
    for metric in metricsToSave
        results_fold[metric] = zeros(Float64, n_folds)
    end

    for i in 1:n_folds
        if showText
            println("Fold ", i, ":")
        end

        # Split the dataset into training and test according to the fold
        trainIdx = findall(kFoldIndices .!= i)
        testIdx = findall(kFoldIndices .== i)
        trainingDatasetFold = (inputs[trainIdx, :], targets[trainIdx])
        testDatasetFold = (inputs[testIdx, :], targets[testIdx])

        # Normalize the datasets
        normalizationParameters = calculateNormalizationParameters(trainingDatasetFold[1], normalizationType)
        performNormalization!(trainingDatasetFold[1], normalizationParameters, normalizationType)
        performNormalization!(testDatasetFold[1], normalizationParameters, normalizationType)

        # Apply PCA if specified
        if applyPCA
            pca = PCA(n_components=pcaThreshold)
            fit!(pca, trainingDatasetFold[1])
            trainingDatasetFold = (
                pca.transform(trainingDatasetFold[1]),
                trainingDatasetFold[2]
            )
            testDatasetFold = (
                pca.transform(testDatasetFold[1]),
                testDatasetFold[2]
            )
        end

        # Create the ensemble model
        ensemble = get_ensemble_model(ensembleType, ensembleHyperParameters, estimators, modelsHyperParameters)
        
        # Train the ensemble
        fit!(ensemble, trainingDatasetFold[1], trainingDatasetFold[2])

        # Predict on the test set
        predictions = ensemble.predict(testDatasetFold[1])

        # Update metrics
        #test_output_onehot = oneHotEncoding(testDatasetFold[2])
        #println(typeof(test_output_onehot))
        update_metrics!(predictions, testDatasetFold[2], metricsToSave, results_fold, i, showText)
    end

    # Print mean and std for each metric
    for metric in metricsToSave
        println("Mean $metric: ", mean(results_fold[metric]), " ± ", std(results_fold[metric]))
    end
    
    return results_fold
end


function trainClassEnsemble(baseEstimator::Symbol, 
    modelsHyperParameters::Dict,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},     
    kFoldIndices::Array{Int64,1},
    NumEstimators::Int=100,
    ensembleType::Symbol=:Voting,
    ensembleHyperParameters::Dict=Dict(),
    metricsToSave::AbstractArray{<:String,1}=["accuracy"],
    showText::Bool = false,
    normalizationType::Symbol=:zeroMean)
    """
    This function trains an ensemble model with the same base model using k-fold cross-validation, receiving the inputs and targets as both real and boolean matrixes, respectively.
    
    Parameters:
        - baseEstimator: The base estimator of the ensemble model.
        - modelsHyperParameters: The hyperparameters of the base model.
        - trainingDataset: Tuple with the inputs and targets of the training dataset, both as real and boolean matrixes, respectively.
        - kFoldIndices: Array with the indices of the k-fold cross-validation.
        - NumEstimators: The number of estimators of the ensemble model.
        - ensembleType: The type of the ensemble model.
        - ensembleHyperParameters: The hyperparameters of the ensemble model.
        - metricsToSave: The metrics to save.
        - showText: A boolean to show the loss in the console.
    
    Returns:
        - results_fold: A list with the results of each fold.
    """
    
    if ensembleType in [:Voting, :Stacking]
        estimators = [baseEstimator for i in 1:NumEstimators];
        modelsHyperParameters = Vector{Dict}([modelsHyperParameters for i in 1:NumEstimators]);
        return trainClassEnsemble(estimators, modelsHyperParameters, trainingDataset, kFoldIndices, ensembleType, ensembleHyperParameters, metricsToSave, showText, normalizationType)
    else
        estimators = [baseEstimator];
        modelsHyperParameters = Vector{Dict}([modelsHyperParameters]);
        ensembleHyperParameters[:n_estimators] = NumEstimators;
        return trainClassEnsemble(estimators, modelsHyperParameters, trainingDataset, kFoldIndices, ensembleType, ensembleHyperParameters, metricsToSave, showText, normalizationType)
    end
    
end