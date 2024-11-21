"""
    This script contains functions to plot the training loss, validation loss, and test loss.
"""

using Plots;
plotly();

function plotTraining(train_loss::AbstractVector{<:Real};
    val_loss::AbstractVector{<:Real}=nothing,
    best_epoch::Int=nothing,
    test_loss::AbstractVector{<:Real}=nothing,
    title::String="Loss Plot", 
    g::Any=nothing,
    model_label::String="Model",
    line_type::Symbol=:solid,
    show::Bool=true,
    size::Tuple{Int,Int}=(800, 600))
    """
    This function plots the training loss, validation loss, and test loss (if provided) in a single plot.
    
    Parameters:
        - train_loss: The training loss values.
        - val_loss: The validation loss values.
        - best_epoch: The epoch with the best validation loss.
        - test_loss: The test loss values.
        - title: The title of the plot.
        - g: The plot object to add the new plot to.
        - model_label: The label of the model.
        - line_type: The line type of the plot.
        - show: Whether to display the plot.
        - size: The size of the plot.
    
    Returns:
        - The plot object in order to add more plots to it if desired or to display it later.
    """

    epochs = 0:length(train_loss)-1

    # Generate the initial plot
    if g == nothing
        g = plot(epochs, train_loss,
            xaxis="Epoch",
            yaxis="Loss value",
            title=title,
            color=:red,
            label="Train from $model_label",
            size=size)
    else
        plot!(g, epochs, train_loss, color=:red, label="Train from $model_label", linestyle=line_type)
    end

    # Add the validation loss if provided
    if val_loss !== nothing
        @assert(length(epochs) == length(val_loss))
        plot!(g, epochs, val_loss, color=:blue, label="Validation from $model_label", linestyle=line_type)
        if best_epoch != nothing
            scatter!([best_epoch], [val_loss[best_epoch+1]], color=:red, label="Best epoch for $model_label")
        end
    end

    # Add the test loss if provided
    if test_loss !== nothing
        @assert(length(epochs) == length(test_loss))
        plot!(g, epochs, test_loss, color=:green, label="Test from $model_label", linestyle=line_type)
    end

    if show
        display(g)
    end

    return g
end