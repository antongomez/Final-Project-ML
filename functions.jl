using Statistics
using Plots
using Random
using Flux.Losses

function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]));
    
    # Second defensive statement, check the number of classes
    numClasses = length(classes);
    @assert(numClasses>1)
    
    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

oneHotEncoding!(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    normalizeMinMax!(copy(dataset), normalizationParameters);
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2})
      normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Remove any atribute that do not have information
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset; 
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset , calculateZeroMeanNormalizationParameters(dataset));   
end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    normalizeZeroMean!(copy(dataset), normalizationParameters);
end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 
   numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)
    accuracy(outputs.>=threshold, targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
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

function trainClassANN(topology::AbstractArray{<:Int,1},      
                    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
                    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 

    (inputs, targets) = dataset;
    
    # This function assumes that each sample is in a row
    # we are going to check the numeber of samples to have same inputs and targets
    @assert(size(inputs,1)==size(targets,1));

    # We define the ANN
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    # Setting up the loss funtion to reduce the error
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    # This vector is going to contain the losses and precission on each training epoch
    trainingLosses = Float32[];

    # Inicialize the counter to 0
    numEpoch = 0;
    # Calcualte the loss without training
    trainingLoss = loss(ann, inputs', targets');
    #  Store this one for checking the evolution.
    push!(trainingLosses, trainingLoss);
    #  and give some feedback on the screen
    println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    # Define the optimazer for the network
    opt_state = Flux.setup(Adam(learningRate), ann);

    # Start the training until it reaches one of the stop critteria
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        # For each epoch, we habve to train and consequently traspose the pattern to have then in columns
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);

        numEpoch += 1;
        # calculate the loss for this epoch
        trainingLoss = loss(ann, inputs', targets');
        # store it
        push!(trainingLosses, trainingLoss);
        # shown it
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    end;

    # return the network and the evolution of the error
    return (ann, trainingLosses);
end;                                        

function trainClassANN(topology::AbstractArray{<:Int,1},      
                    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};      
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),      
                    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    
     trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); 
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);
end;


function holdOut(N::Int, P::Real)
    # Generate a random permutation of the indices
    indices = randperm(N)
    
    # Determine the number of test samples
    test_size = Int(round(P * N))
    
    return (indices[test_size+1:end], indices[1:test_size])
end

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    # Split the test set
    train_val_indices, test_indices = holdOut(N, Ptest)
    
    # Split the remaining patterns into training and validation sets
    train_indices, val_indices = holdOut(size(train_val_indices, 1), Pval / (1 - Ptest))

    return (train_val_indices[train_indices], train_val_indices[val_indices], test_indices)
end

function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false)

    # Extracting inputs and targets from datasets
    (inputs, targets) = trainingDataset;
    (valInputs, valTargets) = validationDataset;
    (testInputs, testTargets) = testDataset;

    # Check that the number of samples matches in inputs and targets
    @assert(size(inputs,1) == size(targets,1));

    # Define the ANN
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    # Define loss function
    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y);

    # Initialize training, validation, and test loss history
    trainingLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];

    # Initialize counters and variables for early stopping
    numEpoch = 0;
    bestValLoss = Inf;
    bestModel = deepcopy(ann);
    epochsWithoutImprovement = 0;

    # Calculate initial losses
    trainingLoss = loss(ann, inputs', targets');
    valLoss = isempty(valInputs) ? 0.0 : loss(ann, valInputs', valTargets');
    testLoss = isempty(testInputs) ? 0.0 : loss(ann, testInputs', testTargets');

    # Store initial losses
    push!(trainingLosses, trainingLoss);
    push!(validationLosses, valLoss);
    push!(testLosses, testLoss);

    # Show initial epoch loss if showText is enabled
    if showText
        println("Epoch ", numEpoch, ": loss: ", trainingLoss, " val_loss: ", valLoss, " test_loss: ", testLoss);
    end

    # Define optimizer
    opt_state = Flux.setup(Adam(learningRate), ann);

    # Training loop
    while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (epochsWithoutImprovement < maxEpochsVal)

        # Train for one epoch
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);

        numEpoch += 1;

        # Calculate losses
        trainingLoss = loss(ann, inputs', targets');
        valLoss = isempty(valInputs) ? 0.0 : loss(ann, valInputs', valTargets');
        testLoss = isempty(testInputs) ? 0.0 : loss(ann, testInputs', testTargets');

        # Store losses
        push!(trainingLosses, trainingLoss);
        push!(validationLosses, valLoss);
        push!(testLosses, testLoss);

        # Show epoch loss if showText is enabled
        if showText
            println("Epoch ", numEpoch, ": loss: ", trainingLoss, " val_loss: ", valLoss, " test_loss: ", testLoss);
        end

        # Early stopping: Check if validation loss improves
        if isempty(valInputs)
            # No validation set, just continue training
            continue;
        elseif valLoss < bestValLoss
            bestValLoss = valLoss;
            bestModel = deepcopy(ann);
            epochsWithoutImprovement = 0;  # Reset the counter
        else
            epochsWithoutImprovement += 1;  # Increment the counter
        end
    end

    # Return the best model if a validation set was used, otherwise the last model
    finalModel = isempty(valInputs) ? ann : bestModel;

    return (finalModel, trainingLosses, validationLosses, testLosses, numEpoch);
end

function trainClassANN(topology::AbstractArray{<:Int,1},  
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
        maxEpochsVal::Int=20, showText::Bool=false)

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


function holdOut(N::Int, P::Real)
    # Generate a random permutation of the indices
    indices = randperm(N)
    
    # Determine the number of test samples
    test_size = Int(round(P * N))
    
    return (indices[test_size+1:end], indices[1:test_size])
end

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    # Split the test set
    train_val_indices, test_indices = holdOut(N, Ptest)
    
    # Split the remaining patterns into training and validation sets
    train_indices, val_indices = holdOut(size(train_val_indices, 1), Pval / (1 - Ptest))

    return (train_val_indices[train_indices], train_val_indices[val_indices], test_indices)
end

function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false)

    # Extracting inputs and targets from datasets
    (inputs, targets) = trainingDataset;
    (valInputs, valTargets) = validationDataset;
    (testInputs, testTargets) = testDataset;

    # Check that the number of samples matches in inputs and targets
    @assert(size(inputs,1) == size(targets,1));

    # Define the ANN
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));

    # Define loss function
    loss(model, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y);

    # Initialize training, validation, and test loss history
    trainingLosses = Float32[];
    validationLosses = Float32[];
    testLosses = Float32[];

    # Initialize counters and variables for early stopping
    numEpoch = 0;
    bestValLoss = Inf;
    bestModel = deepcopy(ann);
    epochsWithoutImprovement = 0;

    # Calculate initial losses
    trainingLoss = loss(ann, inputs', targets');
    valLoss = isempty(valInputs) ? 0.0 : loss(ann, valInputs', valTargets');
    testLoss = isempty(testInputs) ? 0.0 : loss(ann, testInputs', testTargets');

    # Store initial losses
    push!(trainingLosses, trainingLoss);
    push!(validationLosses, valLoss);
    push!(testLosses, testLoss);

    # Show initial epoch loss if showText is enabled
    if showText
        println("Epoch ", numEpoch, ": loss: ", trainingLoss, " val_loss: ", valLoss, " test_loss: ", testLoss);
    end

    # Define optimizer
    opt_state = Flux.setup(Adam(learningRate), ann);

    # Training loop
    while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (epochsWithoutImprovement < maxEpochsVal)

        # Train for one epoch
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);

        numEpoch += 1;

        # Calculate losses
        trainingLoss = loss(ann, inputs', targets');
        valLoss = isempty(valInputs) ? 0.0 : loss(ann, valInputs', valTargets');
        testLoss = isempty(testInputs) ? 0.0 : loss(ann, testInputs', testTargets');

        # Store losses
        push!(trainingLosses, trainingLoss);
        push!(validationLosses, valLoss);
        push!(testLosses, testLoss);

        # Show epoch loss if showText is enabled
        if showText
            println("Epoch ", numEpoch, ": loss: ", trainingLoss, " val_loss: ", valLoss, " test_loss: ", testLoss);
        end

        # Early stopping: Check if validation loss improves
        if isempty(valInputs)
            # No validation set, just continue training
            continue;
        elseif valLoss < bestValLoss
            bestValLoss = valLoss;
            bestModel = deepcopy(ann);
            epochsWithoutImprovement = 0;  # Reset the counter
        else
            epochsWithoutImprovement += 1;  # Increment the counter
        end
    end

    # Return the best model if a validation set was used, otherwise the last model
    finalModel = isempty(valInputs) ? ann : bestModel;

    return (finalModel, trainingLosses, validationLosses, testLosses, numEpoch);
end

function trainClassANN(topology::AbstractArray{<:Int,1},  
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
        maxEpochsVal::Int=20, showText::Bool=false)

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


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    confusion_matrix = [TP FP; FN TN]
    accuracy = (TP + TN) / length(outputs)
    error_rate = (FP + FN) / length(outputs)
    sensitivity = ifelse(TP + FN == 0, 1, TP / (TP + FN))
    specificity = ifelse(TN + FP == 0, 1, TN / (TN + FP))
    ppv = ifelse(TP + FP == 0, 1, TP / (TP + FP))
    npv = ifelse(TN + FN == 0, 1, TN / (TN + FN))
    f_score = ifelse(sensitivity == 0 && ppv == 0, 0, 2 * (ppv * sensitivity) / (ppv + sensitivity))

    return Dict(
        :accuracy => accuracy,
        :error_rate => error_rate,
        :sensitivity => sensitivity,
        :specificity => specificity,
        :ppv => ppv,
        :npv => npv,
        :f_score => f_score,
        :confusion_matrix => confusion_matrix
    )
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real = 0.5)
    # Convert real-valued outputs to boolean based on the threshold
    binary_outputs = outputs .>= threshold
    
    # Call the original confusionMatrix function with boolean outputs
    return confusionMatrix(binary_outputs, targets)
end


# Function for boolean outputs
function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    cm = confusionMatrix(outputs, targets)
    
    println("Confusion Matrix:")
    println(cm[:confusion_matrix])
    println("Accuracy: $(cm[:accuracy])")
    println("Error Rate: $(cm[:error_rate])")
    println("Sensitivity (Recall): $(cm[:sensitivity])")
    println("Specificity: $(cm[:specificity])")
    println("Positive Predictive Value (Precision): $(cm[:ppv])")
    println("Negative Predictive Value: $(cm[:npv])")
    println("F-score: $(cm[:f_score])")
end

# Function for real-valued outputs with optional threshold
function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real = 0.5)
    cm = confusionMatrix(outputs, targets; threshold=threshold)
    
    println("Confusion Matrix:")
    println(cm[:confusion_matrix])
    println("Accuracy: $(cm[:accuracy])")
    println("Error Rate: $(cm[:error_rate])")
    println("Sensitivity (Recall): $(cm[:sensitivity])")
    println("Specificity: $(cm[:specificity])")
    println("Positive Predictive Value (Precision): $(cm[:ppv])")
    println("Negative Predictive Value: $(cm[:npv])")
    println("F-score: $(cm[:f_score])")
end



function oneVSall(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; apply_softmax::Bool=false)
    numInstances, numClasses = size(targets)
    outputs = Array{Float32,2}(undef, numInstances, numClasses)

    # Simulate binary classifier output for each class
    for numClass in 1:numClasses
        model = fit(inputs, targets[:,[numClass]])
        outputs[:,numClass] .= model(inputs)
    end

    # Optionally apply softmax to outputs
    if apply_softmax
        outputs = softmax(outputs')' 
    end    

    # Convert the output to a boolean matrix, ensuring only one 'true' per row
    vmax = maximum(outputs, dims=2)
    outputs_bool = (outputs .== vmax)
    
    # Resolve ties by keeping only the first occurrence of the maximum in each row
    for i in 1:numInstances
        max_indices = findall(outputs_bool[i, :])
        if length(max_indices) > 1
            # Keep only the first occurrence of the maximum value
            outputs_bool[i, :] .= false
            outputs_bool[i, max_indices[1]] = true
        end
    end

    return outputs_bool, outputs
end


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    numClasses = size(outputs, 2)
    if numClasses != size(targets, 2) || numClasses == 2
        error("Invalid number of columns. Outputs and targets must have more than 2 classes and the same number of columns.")
    end

    if numClasses == 1
        return confusionMatrix(outputs[:, 1], targets[:, 1])
    end

    sensitivity = zeros(Float64, numClasses)
    specificity = zeros(Float64, numClasses)
    ppv = zeros(Float64, numClasses)
    npv = zeros(Float64, numClasses)
    f1_score = zeros(Float64, numClasses)
    valid_classes = 0

    for class in 1:numClasses
        if any(targets[:, class])
            cm = confusionMatrix(outputs[:, class], targets[:, class])
            sensitivity[class] = cm[:sensitivity]
            specificity[class] = cm[:specificity]
            ppv[class] = cm[:ppv]
            npv[class] = cm[:npv]
            f1_score[class] = cm[:f_score]
            valid_classes += 1
        end
    end

    conf_matrix = zeros(Int, numClasses, numClasses)
    for i in 1:numClasses
        for j in 1:numClasses
            conf_matrix[i, j] = sum(outputs[:, i] .& targets[:, j])
        end
    end

    if weighted
        class_weights = sum(targets, dims=1) ./ sum(targets)
        agg_sensitivity = sum(sensitivity .* class_weights)
        agg_specificity = sum(specificity .* class_weights)
        agg_ppv = sum(ppv .* class_weights)
        agg_npv = sum(npv .* class_weights)
        agg_f1_score = sum(f1_score .* class_weights)
    else
        agg_sensitivity = sum(sensitivity) / valid_classes
        agg_specificity = sum(specificity) / valid_classes
        agg_ppv = sum(ppv) / valid_classes
        agg_npv = sum(npv) / valid_classes
        agg_f1_score = sum(f1_score) / valid_classes
    end

    accuracy_val = accuracy(outputs, targets)
    error_rate = 1 - accuracy_val

    return Dict(
        :accuracy => accuracy_val,
        :error_rate => error_rate,
        :sensitivity => agg_sensitivity,
        :specificity => agg_specificity,
        :ppv => agg_ppv,
        :npv => agg_npv,
        :f1_score => agg_f1_score,
        :confusion_matrix => conf_matrix
    )
end


function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Convert real-valued outputs to boolean using classifyOutputs
    bool_outputs = classifyOutputs(outputs)
    
    # Call the previously defined confusionMatrix for boolean outputs and targets
    return confusionMatrix(bool_outputs, targets; weighted=weighted)
end


function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Defensive line to ensure all output classes are in the target classes
    @assert (all([in(output, unique(targets)) for output in outputs])) "All output classes must be included in the target classes"
    
    # Calculate the unique classes from both outputs and targets
    class_vector = unique(vcat(outputs, targets))
    
    # Encode both outputs and targets using oneHotEncoding
    bool_outputs = oneHotEncoding(outputs, class_vector)
    bool_targets = oneHotEncoding(targets, class_vector)
    
    # Call the previously defined confusionMatrix function for boolean matrices
    return confusionMatrix(bool_outputs, bool_targets; weighted=weighted)
end







using Random

function crossvalidation(N::Int64, k::Int64)
    # Create a vector with k sorted elements (from 1 to k)
    folds = collect(1:k)
    
    # Repeat the vector enough times to cover N elements
    repeated_folds = repeat(folds, ceil(Int, N / k))
    
    # Take the first N values from the repeated vector
    indices = repeated_folds[1:N]
    
    # Shuffle the indices
    shuffle!(indices)
    
    return indices
end


function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    N = size(targets, 1)  # Number of rows in the target matrix
    num_classes = size(targets, 2)  # Number of columns (classes) in the target matrix
    
    # Create an empty index vector of length N
    indices = zeros(Int64, N)
    
    # Iterate over classes and assign patterns to folds
    for class in 1:num_classes
        # Get indices of elements that belong to the current class
        class_indices = findall(targets[:, class])
        num_elements_in_class = length(class_indices)
        
        # Ensure that each class has at least k patterns
        if num_elements_in_class < k
            error("Class $class has fewer patterns ($num_elements_in_class) than the number of folds k=$k")
        end
        
        # Call the crossvalidation function developed earlier for the current class
        fold_assignments = crossvalidation(num_elements_in_class, k)
        
        # Update the index vector for rows corresponding to this class
        indices[class_indices] .= fold_assignments
    end
    
    return indices
end


function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    """
    The following lines handle the crossvalidation without using oneHotEncoding:
    
    N = length(targets)
    # Create an empty index vector of length N
    indices = zeros(Int64, N)
    # Find unique categories in the target vector
    num_classes = unique(targets)

    Also inside the for loop, we need to change the way we get the class_indices:
    class_indices = findall(x -> x == class, targets)
    """ 
    
    N = length(targets)
    indices = zeros(Int64, N)
    num_classes = unique(targets)

    for class in num_classes
        # Get indices of elements that belong to the current class
        class_indices = findall(targets .== class)
        num_elements_in_class = length(class_indices)

        # Ensure that each category has at least k patterns
        if num_elements_in_class < k
            error("Class '$class' has fewer patterns ($num_elements_in_class) than the number of folds k=$k")
        end

        # Assign folds to elements of the current class
        fold_assignments = crossvalidation(num_elements_in_class, k)
        
        # Update the index vector for the current class
        indices[class_indices] .= fold_assignments
    end

    return indices
end


function trainClassANN(
    topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
    kFoldIndices::Array{Int64,1}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Float64=0.01, 
    repetitionsTraining::Int=1, validationRatio::Real=0.0, maxEpochsVal::Int=20
)

    (inputs, targets) = trainingDataset
    N = size(inputs, 1)
    k = maximum(kFoldIndices)

    # Initialize vectors for metrics for each fold
    testAccuracies = zeros(Float32, k)
    testSensitivities = zeros(Float32, k)
    testSpecificities = zeros(Float32, k)
    testF1Scores = zeros(Float32, k)

    for fold in 1:k
        # Create training and test sets based on kFoldIndices
        testIdx = findall(kFoldIndices .== fold)
        trainIdx = findall(kFoldIndices .!= fold)
        
        trainInputs, trainTargets = inputs[trainIdx, :], targets[trainIdx, :]
        testInputs, testTargets = inputs[testIdx, :], targets[testIdx, :]

        # Initialize metrics for each repetition within this fold
        foldAccuracies, foldSensitivities, foldSpecificities, foldF1Scores = Float32[], Float32[], Float32[], Float32[]

        for rep in 1:repetitionsTraining
            if validationRatio > 0.0
                # If validationRatio > 0.0, split training into training and validation subsets
                trainSubsetIdx, valIdx = holdOut(size(trainInputs, 1), validationRatio)
                valInputs, valTargets = trainInputs[valIdx, :], trainTargets[valIdx, :]
                trainSubsetInputs, trainSubsetTargets = trainInputs[trainSubsetIdx, :], trainTargets[trainSubsetIdx, :]
                
                # Call trainClassANN with validation data
                model, _, _, _, _ = trainClassANN(
                    topology, (trainSubsetInputs, trainSubsetTargets);
                    validationDataset = (valInputs, valTargets),
                    testDataset = (testInputs, testTargets),
                    transferFunctions = transferFunctions,
                    maxEpochs = maxEpochs,
                    minLoss = minLoss,
                    learningRate = learningRate,
                    maxEpochsVal = maxEpochsVal
                )
            else
                # Call trainClassANN without validation data
                model, _, _, _, _ = trainClassANN(
                    topology, (trainInputs, trainTargets);
                    testDataset = (testInputs, testTargets),
                    transferFunctions = transferFunctions,
                    maxEpochs = maxEpochs,
                    minLoss = minLoss,
                    learningRate = learningRate,
                    maxEpochsVal = maxEpochsVal
                )
            end
            
            # Predict on the test set and calculate metrics
            predictedTestOutputs = Matrix(model(testInputs')') .>= 0.5
            cm = confusionMatrix(predictedTestOutputs, testTargets)
            accuracy = cm[:accuracy]
            sensitivity = cm[:sensitivity]
            specificity = cm[:specificity]
            f1_score = cm[:f1_score]

            accuracy = isnan(accuracy) ? 0.0 : accuracy
            sensitivity = isnan(sensitivity) ? 0.0 : sensitivity
            specificity = isnan(specificity) ? 0.0 : specificity
            f1_score = isnan(f1_score) ? 0.0 : f1_score
            
            # Store the metrics for this repetition
            push!(foldAccuracies, accuracy)
            push!(foldSensitivities, sensitivity)
            push!(foldSpecificities, specificity)
            push!(foldF1Scores, f1_score)
        end
        
        # Average the metrics across repetitions for this fold
        testAccuracies[fold] = mean(foldAccuracies)
        testSensitivities[fold] = mean(foldSensitivities)
        testSpecificities[fold] = mean(foldSpecificities)
        testF1Scores[fold] = mean(foldF1Scores)

        println("Fold $fold - Accuracies: $foldAccuracies, Sensitivities: $foldSensitivities, Specificities: $foldSpecificities, F1 Scores: $foldF1Scores")
    end

    # Final output with means and standard deviations across folds
    println("Final results across folds:")
    println("Accuracy: $(mean(testAccuracies)) ± $(std(testAccuracies))")
    println("Sensitivity: $(mean(testSensitivities)) ± $(std(testSensitivities))")
    println("Specificity: $(mean(testSpecificities)) ± $(std(testSpecificities))")
    println("F1 Score: $(mean(testF1Scores)) ± $(std(testF1Scores))")

    return (mean(testAccuracies), std(testAccuracies),
            mean(testSensitivities), std(testSensitivities),
            mean(testSpecificities), std(testSpecificities),
            mean(testF1Scores), std(testF1Scores))
end


function trainClassANN(
    topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, 
    kFoldIndices::Array{Int64,1}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    repetitionsTraining::Int=1, validationRatio::Real=0.0, maxEpochsVal::Int=20
)

    (trainingInputs, trainingTargets) = trainingDataset

    # Reshape the trainingTargets to work as a 2D matrix (length, 1) if it’s a 1D vector
    reshapedTargets = reshape(trainingTargets, length(trainingTargets), 1)

    return trainClassANN(
        topology, (trainingInputs, reshapedTargets), kFoldIndices;
        transferFunctions = transferFunctions,
        maxEpochs = maxEpochs,
        minLoss = minLoss,
        learningRate = learningRate,
        repetitionsTraining = repetitionsTraining,
        validationRatio = validationRatio,
        maxEpochsVal = maxEpochsVal
    )
end


function modelCrossValidation(
    modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1}
)
    accuracies, sensitivities, specificities, f1Scores = Float32[], Float32[], Float32[], Float32[]
    num_folds = maximum(crossValidationIndices)

    for fold in 1:num_folds
        # Separate training and validation data for this fold
        trainIdx = findall(crossValidationIndices .!= fold)
        valIdx = findall(crossValidationIndices .== fold)
        
        trainInputs, trainTargets = inputs[trainIdx, :], targets[trainIdx]
        valInputs, valTargets = inputs[valIdx, :], targets[valIdx]

        if modelType == :ANN
            # Extract parameters from modelHyperparameters
            topology = get(modelHyperparameters, :topology, [4,3])
            transferFunctions = get(modelHyperparameters, :transferFunctions, fill(σ, length(topology)))
            maxEpochs = get(modelHyperparameters, :max_epochs, 100)
            learningRate = get(modelHyperparameters, :learning_rate, 0.01)
            validationRatio = get(modelHyperparameters, :validation_ratio, 0.0)
            trainTargets = oneHotEncoding(trainTargets)

            # Train the ANN model with the specified parameters
            accuracy, _, sensitivity, _, specificity, _, f1_score, _ = trainClassANN(
                topology, 
                (trainInputs, trainTargets), 
                crossValidationIndices[trainIdx];
                transferFunctions = transferFunctions,
                maxEpochs = maxEpochs,
                learningRate = learningRate,
                repetitionsTraining = 10,
                validationRatio = validationRatio,
            )

            accuracy = isnan(accuracy) ? 0.0 : accuracy
            sensitivity = isnan(sensitivity) ? 0.0 : sensitivity
            specificity = isnan(specificity) ? 0.0 : specificity
            f1_score = isnan(f1_score) ? 0.0 : f1_score

        elseif modelType == :SVM
            model = SVC(; modelHyperparameters...)
            fit!(model, trainInputs, trainTargets)
            predictions = predict(model, valInputs)

        elseif modelType == :DecisionTree
            model = DecisionTreeClassifier(; modelHyperparameters...)
            fit!(model, trainInputs, trainTargets)
            predictions = predict(model, valInputs)
        
        elseif modelType == :kNN
            model = KNeighborsClassifier(; modelHyperparameters...)
            fit!(model, trainInputs, trainTargets)
            predictions = predict(model, valInputs)
        
        else
            error("Unknown model type")
        end

        if modelType != :ANN
            # Calculate and store metrics
            cm = confusionMatrix(predictions, valTargets)
            accuracy = cm[:accuracy]
            sensitivity = cm[:sensitivity]
            specificity = cm[:specificity]
            f1_score = cm[:f1_score]

        end
        
        push!(accuracies, accuracy)
        push!(sensitivities, sensitivity)
        push!(specificities, specificity)
        push!(f1Scores, f1_score)

    end
    
    # Calculate and return average metrics across folds
    return Dict(
        :accuracy => mean(accuracies),
        :sensitivity => mean(sensitivities),
        :specificity => mean(specificities),
        :f1Score => mean(f1Scores)
    )
end


# Split dataset into training and testing sets
function split_data(input_data, output_data, train_ratio = 0.8)
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


function trainClassEnsemble(
    estimators::AbstractArray{Symbol,1}, 
    modelsHyperParameters::AbstractArray{Dict,1},     
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},    
    kFoldIndices::AbstractArray{Int64,1}
)

    # Unpack training data
    input_data, output_data = trainingDataset
    k = maximum(kFoldIndices)

    # Initialize vectors for metrics for each fold
    testAccuracies = zeros(Float32, k)
    testSensitivities = zeros(Float32, k)
    testSpecificities = zeros(Float32, k)
    testF1Scores = zeros(Float32, k)

    for fold in 1:k
        # Create training and test sets based on kFoldIndices
        testIdx = findall(kFoldIndices .== fold)
        trainIdx = findall(kFoldIndices .!= fold)

        # Split data into training and testing sets
        train_input, test_input = input_data[trainIdx, :], input_data[testIdx, :]
        train_output, test_output = output_data[trainIdx, :], output_data[testIdx, :]
        
        # Combine models in an ensemble
        ensemble_model = VotingClassifier(
            estimators = [
                begin
                    # Pass the hyperparameters directly, including validation_fraction if present
                    ("model_$i", eval(estimator)(; hyperparams...))
                end
                for (i, (estimator, hyperparams)) in enumerate(zip(estimators, modelsHyperParameters))
            ],
            voting = "hard"
        )
        
        # Train the ensemble on the training set
        fit!(ensemble_model, train_input, vec(train_output))

        # Predict and calculate metrics
        predictedTestOutputs = predict(ensemble_model, test_input) .>= 0.5
        cm = confusionMatrix(predictedTestOutputs, vec(test_output))

        # Extract metrics
        accuracy = cm[:accuracy]
        sensitivity = cm[:sensitivity]
        specificity = cm[:specificity]
        f1_score = cm[:f_score]

        # Handle NaNs in metrics
        accuracy = isnan(accuracy) ? 0.0 : accuracy
        sensitivity = isnan(sensitivity) ? 0.0 : sensitivity
        specificity = isnan(specificity) ? 0.0 : specificity
        f1_score = isnan(f1_score) ? 0.0 : f1_score

        # Store metrics for the fold
        testAccuracies[fold] = accuracy
        testSensitivities[fold] = sensitivity
        testSpecificities[fold] = specificity
        testF1Scores[fold] = f1_score

        println("Fold $fold - Accuracy: $accuracy, Sensitivity: $sensitivity, Specificity: $specificity, F1 Score: $f1_score")
    end

    # Final output with means and standard deviations across folds
    println("Final results across folds:")
    println("Accuracy: $(mean(testAccuracies)) ± $(std(testAccuracies))")
    println("Sensitivity: $(mean(testSensitivities)) ± $(std(testSensitivities))")
    println("Specificity: $(mean(testSpecificities)) ± $(std(testSpecificities))")
    println("F1 Score: $(mean(testF1Scores)) ± $(std(testF1Scores))")

    return (mean(testAccuracies), std(testAccuracies),
            mean(testSensitivities), std(testSensitivities),
            mean(testSpecificities), std(testSpecificities),
            mean(testF1Scores), std(testF1Scores))
end


function trainClassEnsemble(
    baseEstimator::Symbol, 
    modelsHyperParameters::Dict,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},     
    kFoldIndices::AbstractArray{Int64,1},
    NumEstimators::Int = 100
)

    # Create repeated estimators and hyperparameters
    estimators = fill(baseEstimator, NumEstimators)
    hyperParamsArray = fill(modelsHyperParameters, NumEstimators)

    # Call the original trainClassEnsemble function
    return trainClassEnsemble(
        estimators, 
        hyperParamsArray, 
        trainingDataset, 
        kFoldIndices
    )
end