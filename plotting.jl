"""
    This script contains functions to plot the training loss, validation loss, and test loss.
"""

using Plots;
# plotly();
using DataFrames, PrettyTables;

function plotTraining(train_loss::AbstractVector{<:Real};
    val_loss::AbstractVector{<:Real}=nothing,
    best_epoch::Int=nothing,
    test_loss::AbstractVector{<:Real}=nothing,
    title::String="Loss Plot", 
    g::Any=nothing,
    model_label::String="Model",
    line_type::Symbol=:solid,
    show::Bool=true,
    size::Tuple{Int,Int}=(800, 600),
    ylim::Tuple{<:Real,<:Real}=(0.0, 1.0))
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
        - ylim: Tuple specifying the y-axis limits (default is (0.0, 1.0)).
    
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
            size=size,
            ylim=ylim)
    else
        plot!(g, epochs, train_loss, color=:red, label="Train from $model_label", linestyle=line_type, ylim=ylim)
    end

    # Add the validation loss if provided
    if val_loss !== nothing
        @assert(length(epochs) == length(val_loss))
        plot!(g, epochs, val_loss, color=:blue, label="Validation from $model_label", linestyle=line_type, ylim=ylim)
        if best_epoch != nothing
            scatter!([best_epoch], [val_loss[best_epoch+1]], color=:red, label="Best epoch for $model_label", ylim=ylim)
        end
    end

    # Add the test loss if provided
    if test_loss !== nothing
        @assert(length(epochs) == length(test_loss))
        plot!(g, epochs, test_loss, color=:green, label="Test from $model_label", linestyle=line_type, ylim=ylim)
    end

    if show
        display(g)
    end

    return g
end

function aggregateMetrics(
    loaded_obj::Dict{Symbol, Dict{String, Any}};
    metrics::Vector{Symbol} = [:accuracy, :precision, :recall, :f1_score],
    ensemble::Bool = false
)
    model_names = []
    metric_means = Dict(metric => [] for metric in metrics)
    metric_stds = Dict(metric => [] for metric in metrics)
    metric_maxes = Dict(metric => [] for metric in metrics)

    for (algorithm, results) in loaded_obj
        push!(model_names, string(algorithm))
        general_results = results["general_results"]

        for metric in metrics
            if ensemble
                push!(metric_means[metric], mean(general_results[metric]))
                push!(metric_stds[metric], std(general_results[metric]))
                
                # Compute maximum across all ensemble results
                push!(metric_maxes[metric], maximum(general_results[metric]))
            else
                metric_values = [mean(general_result[metric]) for general_result in general_results]
                push!(metric_means[metric], mean(metric_values))
                push!(metric_stds[metric], std(metric_values))

                # Compute maximum across all trained models
                metric_values_max = [maximum(general_result[metric]) for general_result in general_results]
                push!(metric_maxes[metric], maximum(metric_values_max))
            end
        end
    end

    return model_names, metrics, metric_means, metric_stds, metric_maxes
end


# Also the metrics per class
function aggregateMetrics(
    loaded_obj::Dict{Symbol, Dict{String, Any}},
    numClasses::Int64; 
    metrics::Vector{Symbol} = [:accuracy, :precision, :recall, :f1_score],
    ensemble::Bool = false
)
    model_names = []
    metric_means = Dict(metric => [] for metric in metrics)
    metric_stds = Dict(metric => [] for metric in metrics)
    metric_means_class = [Dict(metric => [] for metric in metrics) for _ in 1:numClasses]
    metric_stds_class = [Dict(metric => [] for metric in metrics) for _ in 1:numClasses]

    metric_maxes = Dict(metric => [] for metric in metrics)
    metric_maxes_class = [Dict(metric => [] for metric in metrics) for _ in 1:numClasses]

    for (algorithm, results) in loaded_obj
        push!(model_names, string(algorithm))
        general_results = results["general_results"]
        class_results = results["class_results"]

        for metric in metrics
            if ensemble
                push!(metric_means[metric], mean(general_results[metric]))
                push!(metric_stds[metric], std(general_results[metric]))
                for i in 1:numClasses
                    push!(metric_means_class[i][metric], mean(class_results[i][metric]))
                    push!(metric_stds_class[i][metric], std(class_results[i][metric]))
                end

                # Compute maximum for general metrics
                push!(metric_maxes[metric], maximum(general_results[metric]))

                # Compute maximum for per-class metrics
                for i in 1:numClasses
                    push!(metric_maxes_class[i][metric], maximum(class_results[i][metric]))
                end
            else
                metric_values = [mean(general_result[metric]) for general_result in general_results]
                push!(metric_means[metric], mean(metric_values))
                push!(metric_stds[metric], std(metric_values))
                for i in 1:numClasses
                    metric_values = [mean(result[i][metric]) for result in class_results]
                    push!(metric_means_class[i][metric], mean(metric_values))
                    push!(metric_stds_class[i][metric], std(metric_values))
                end

                # Compute maximum for general metrics
                metric_values_max = [maximum(general_result[metric]) for general_result in general_results]
                push!(metric_maxes[metric], maximum(metric_values_max))

                # Compute maximum for per-class metrics
                for i in 1:numClasses
                    metric_values_max = [maximum(result[i][metric]) for result in class_results]
                    push!(metric_maxes_class[i][metric], maximum(metric_values_max))
                end
            end
        end
    end

    return model_names, metrics, metric_means, metric_stds, metric_means_class, metric_stds_class, metric_maxes, metric_maxes_class
end

function plotMetricsPerAlgorithm(
    loaded_obj::Dict{Symbol, Dict{String, Any}}; 
    output_dir::String = "./plots/",
    metrics::Vector{Symbol} = [:accuracy, :precision, :recall, :f1_score],
    size::Tuple{Int, Int} = (1200, 600),
    ylim::Tuple{<:Real,<:Real}=(0.0, 1.0)
)
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    for (algorithm, results) in loaded_obj
        num_trained_models = results["num_trained_models"]
        general_results = results["general_results"]
        parameters_names = keys(results["parameters"])
        parameters = results["parameters"]

        save_folder = joinpath(output_dir, string(algorithm))
        if !isdir(save_folder)
            mkdir(save_folder)
        end

        param_labels = [join([string(param, ": ", parameters[param][i]) for param in parameters_names], ", ") for i in 1:num_trained_models]

        for metric in metrics
            mean_values = [mean(general_results[i][metric]) for i in 1:num_trained_models]
            std_values = [std(general_results[i][metric]) for i in 1:num_trained_models]

            lower_limit = minimum(mean_values .- std_values)
            upper_limit = maximum(mean_values .+ std_values)

            ylim = (
                max(lower_limit - 0.15, 0),
                min(upper_limit + 0.15, 1)
            )

            # Bar plot
            bar_plot = bar(
                1:num_trained_models,
                mean_values,
                yerror=std_values,
                ylabel=string(metric, " (mean ± std)"),
                title="Performance of $(metric) for $(algorithm)",
                legend=false,
                grid=true,
                xticks=(1:num_trained_models, param_labels),
                size=size,
                ylim=ylim
            )
            savefig(joinpath(save_folder, string(metric, "_performance_bar.png")))

            # Line plot
            line_plot = plot(
                1:num_trained_models,
                mean_values,
                ribbon=std_values,
                xlabel="Hyperparameter Combination",
                ylabel=string(metric, " (mean ± std)"),
                title="Performance of $(metric) for $(algorithm)",
                label="Mean $(metric)",
                grid=true,
                xticks=(1:num_trained_models, param_labels),
                lw=2,
                markershape=:circle,
                size=size,
                ylim=ylim
            )
            savefig(joinpath(save_folder, string(metric, "_performance_line.png")))
            println("Saved plots for $(algorithm) and $(metric).")
        end
    end
end

function plotMetricsPerClassAlgorithm(
    loaded_obj::Dict{Symbol, Dict{String, Any}},
    numClasses::Int64; 
    output_dir::String = "./plots/",
    metrics::Vector{Symbol} = [:accuracy, :precision, :recall, :f1_score],
    size::Tuple{Int, Int} = (1200, 600),
    ylim::Tuple{<:Real,<:Real}=(0.0, 1.0)
)
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    for (algorithm, results) in loaded_obj
        num_trained_models = results["num_trained_models"]
        class_results = results["class_results"]
        parameters_names = keys(results["parameters"])
        parameters = results["parameters"]

        save_folder = joinpath(output_dir, string(algorithm))
        if !isdir(save_folder)
            mkdir(save_folder)
        end

        param_labels = [join([string(param, ": ", parameters[param][i]) for param in parameters_names], ", ") for i in 1:num_trained_models]

        for i in 1:numClasses
            for metric in metrics
                mean_values = [mean(class_results[j][i][metric]) for j in 1:num_trained_models]
                std_values = [std(class_results[j][i][metric]) for j in 1:num_trained_models]

                lower_limit = minimum(mean_values .- std_values)
                upper_limit = maximum(mean_values .+ std_values)

                ylim = (
                    max(lower_limit - 0.15, 0),
                    min(upper_limit + 0.15, 1)
                )

                # Bar plot
                bar_plot = bar(
                    1:num_trained_models,
                    mean_values,
                    yerror=std_values,
                    ylabel=string(metric, " (mean ± std)"),
                    title="Performance of $(metric) for $(algorithm) on Class $(i)",
                    legend=false,
                    grid=true,
                    xticks=(1:num_trained_models, param_labels),
                    size=size,
                    ylim=ylim
                )
                savefig(joinpath(save_folder, string(metric, "_performance_class_", i, "_bar.png")))

                # Line plot
                line_plot = plot(
                    1:num_trained_models,
                    mean_values,
                    ribbon=std_values,
                    xlabel="Hyperparameter Combination",
                    ylabel=string(metric, " (mean ± std)"),
                    title="Performance of $(metric) for $(algorithm) on Class $(i)",
                    label="Mean $(metric)",
                    grid=true,
                    xticks=(1:num_trained_models, param_labels),
                    lw=2,
                    markershape=:circle,
                    size=size,
                    ylim=ylim,
                )
                savefig(joinpath(save_folder, string(metric, "_performance_class_", i, "_line.png")))
                println("Saved plots for $(algorithm) and $(metric) on Class $(i).")
            end
        end
    end
end


function generateAlgorithmTables(
    loaded_obj::Dict{Symbol, Dict{String, Any}}; 
    sort_by::Symbol = :Accuracy,
    rev::Bool = true,
    output_dir::String = "./tables/"
)
    # Ensure output directory exists
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    for (algorithm, results) in loaded_obj
        num_trained_models = results["num_trained_models"]
        general_results = results["general_results"]
        parameters_names = keys(results["parameters"])
        parameters = results["parameters"]

        # Prepare labels for hyperparameter configurations
        param_labels = [join([string(param, ": ", parameters[param][i]) for param in parameters_names], ", ") for i in 1:num_trained_models]

        # Initialize arrays to store metrics
        configurations = String[]
        accuracies = Float64[]
        precisions = Float64[]
        recalls = Float64[]
        f1_scores = Float64[]

        # Populate the arrays
        for i in 1:num_trained_models
            try
                push!(configurations, param_labels[i])
                push!(accuracies, maximum(general_results[i][:accuracy]))
                push!(precisions, maximum(general_results[i][:precision]))
                push!(recalls, maximum(general_results[i][:recall]))
                push!(f1_scores, maximum(general_results[i][:f1_score]))
            catch e
                println("Skipping configuration $(param_labels[i]) due to missing or invalid data.")
            end
        end

        # Check if data is complete
        if isempty(configurations) || isempty(accuracies)
            println("No valid data for algorithm: $(algorithm). Skipping...")
            continue
        end

        # Create DataFrame for the current algorithm
        model_table = DataFrame(
            Configuration = configurations,
            Accuracy = accuracies,
            Precision = precisions,
            Recall = recalls,
            F1_Score = f1_scores
        )

        # Sort table by the specified metric
        sorted_table = sort(model_table, sort_by, rev=rev)

        # Show the table
        println("\nComparison of Hyperparameter Configurations for $(algorithm) (Sorted by $(string(sort_by))):")
        pretty_table(sorted_table, header=["Configuration", "Accuracy", "Precision", "Recall", "F1-Score"])

        # Save the table as text
        txt_file = joinpath(output_dir, "$(algorithm)_hyperparameter_configurations.txt")
        open(txt_file, "w") do io
            println(io, "\nComparison of Hyperparameter Configurations for $(algorithm) (Sorted by $(string(sort_by))):")
            pretty_table(io, sorted_table, header=["Configuration", "Accuracy", "Precision", "Recall", "F1-Score"])
        end

        println("Results for $(algorithm) saved to $(output_dir).")
    end
end

function generateClassAlgorithmTables(
    loaded_obj::Dict{Symbol, Dict{String, Any}},
    numClasses::Int64; 
    sort_by::Symbol = :Accuracy,
    rev::Bool = true,
    output_dir::String = "./tables/"
)
    # Ensure output directory exists
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    for (algorithm, results) in loaded_obj
        num_trained_models = results["num_trained_models"]
        class_results = results["class_results"]
        parameters_names = keys(results["parameters"])
        parameters = results["parameters"]

        # Prepare labels for hyperparameter configurations
        param_labels = [join([string(param, ": ", parameters[param][i]) for param in parameters_names], ", ") for i in 1:num_trained_models]

        for i in 1:numClasses
            # Initialize arrays to store metrics
            configurations = String[]
            accuracies = Float64[]
            precisions = Float64[]
            recalls = Float64[]
            f1_scores = Float64[]

            # Populate the arrays
            for j in 1:num_trained_models
                try
                    push!(configurations, param_labels[j])
                    push!(accuracies, mean(class_results[j][i][:accuracy]))
                    push!(precisions, mean(class_results[j][i][:precision]))
                    push!(recalls, mean(class_results[j][i][:recall]))
                    push!(f1_scores, mean(class_results[j][i][:f1_score]))
                catch e
                    println("Skipping configuration $(param_labels[j]) for Class $(i) due to missing or invalid data.")
                end
            end

            # Check if data is complete
            if isempty(configurations) || isempty(accuracies)
                println("No valid data for algorithm: $(algorithm) on Class $(i). Skipping...")
                continue
            end

            # Create DataFrame for the current algorithm
            model_table = DataFrame(
                Configuration = configurations,
                Accuracy = accuracies,
                Precision = precisions,
                Recall = recalls,
                F1_Score = f1_scores
            )

            # Sort table by the specified metric
            sorted_table = sort(model_table, sort_by, rev=rev)

            # Show the table
            println("\nComparison of Hyperparameter Configurations for $(algorithm) on Class $(i) (Sorted by $(string(sort_by))):")
            pretty_table(sorted_table, header=["Configuration", "Accuracy", "Precision", "Recall", "F1-Score"])
            
            # Save the table as text
            txt_file = joinpath(output_dir, "$(algorithm)_hyperparameter_configurations_class_$(i).txt")
            open(txt_file, "w") do io
                println(io, "\nComparison of Hyperparameter Configurations for $(algorithm) on Class $(i) (Sorted by $(string(sort_by))):")
                pretty_table(io, sorted_table, header=["Configuration", "Accuracy", "Precision", "Recall", "F1-Score"])
            end

            println("Results for $(algorithm) on Class $(i) saved to $(output_dir).")
        end
    end
end

function plotCombinedMetrics(
    model_names::Vector{Any},
    metrics::Vector{Symbol},
    metric_means::Dict{Symbol, Vector{Any}},
    metric_stds::Dict{Symbol, Vector{Any}};
    output_dir::String = "./plots/",
    size::Tuple{Int, Int} = (800, 600),
    show::Bool = true,
    ylim::Tuple{<:Real,<:Real}=(0.0, 1.0)
)
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    for metric in metrics

        lower_limit = minimum(metric_means[metric] .- metric_stds[metric])
        upper_limit = maximum(metric_means[metric] .+ metric_stds[metric])

        ylim = (
            max(lower_limit - 0.15, 0),
            min(upper_limit + 0.15, 1)
        )

        # Bar plot
        bar_plot = bar(
            model_names,
            metric_means[metric],
            yerror=metric_stds[metric],
            xlabel="Model",
            ylabel=string(metric, " (mean ± std)"),
            title="Comparison of Models based on $(metric)",
            legend=false,
            grid=true,
            ylim=ylim
        )

        # Line plot
        line_plot = plot(
            model_names,
            metric_means[metric],
            ribbon=metric_stds[metric],
            xlabel="Model",
            ylabel=string(metric, " (mean ± std)"),
            title="Trends in $(metric) Across Models",
            label=string("Mean ", metric),
            lw=2,
            grid=true,
            ylim=ylim
        )

        # Combined plot
        combined_plot = plot(
            bar_plot,
            line_plot,
            layout=(2, 1),
            size=size
        )

        savefig(joinpath(output_dir, "combined_$(metric)_plots.png"))
        if show
            display(combined_plot)
        end
    end
end

function plotCombinedMetrics(
    model_names::Vector{Any},
    numClasses::Int64,
    metrics::Vector{Symbol},
    metric_means_class::Vector{Dict{Symbol, Vector{Any}}},
    metric_stds_class::Vector{Dict{Symbol, Vector{Any}}};
    output_dir::String = "./plots/",
    size::Tuple{Int, Int} = (800, 600),
    show::Bool = true,
    ylim::Tuple{<:Real,<:Real}=(0.0, 1.0)
)
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    for i in 1:numClasses
        for metric in metrics

            lower_limit = minimum(metric_means_class[i][metric] .- metric_stds_class[i][metric])
            upper_limit = maximum(metric_means_class[i][metric] .+ metric_stds_class[i][metric])

            ylim = (
                max(lower_limit - 0.15, 0),
                min(upper_limit + 0.15, 1)
            )

            # Bar plot
            bar_plot = bar(
                model_names,
                metric_means_class[i][metric],
                yerror=metric_stds_class[i][metric],
                xlabel="Model",
                ylabel=string(metric, " (mean ± std)"),
                title="Comparison of Models based on $(metric) for Class $(i)",
                legend=false,
                grid=true,
                ylim = ylim
            )

            # Line plot
            line_plot = plot(
                model_names,
                metric_means_class[i][metric],
                ribbon=metric_stds_class[i][metric],
                xlabel="Model",
                ylabel=string(metric, " (mean ± std)"),
                title="Trends in $(metric) Across Models for Class $(i)",
                label=string("Mean ", metric),
                lw=2,
                grid=true,
                ylim=ylim
            )

            # Combined plot
            combined_plot = plot(
                bar_plot,
                line_plot,
                layout=(2, 1),
                size=size
            )

            savefig(joinpath(output_dir, "combined_$(metric)_plots_class_$(i).png"))
            if show
                display(combined_plot)
            end
        end
    end
end


function generateComparisonTable(
    model_names::Vector{Any},
    metrics::Vector{Symbol},
    metric_maxes::Dict{Symbol, Vector{Any}}; # Use `;` to define keyword arguments
    sort_by::Symbol = :Accuracy,
    rev::Bool = true
)
    comparison_table = DataFrame(Model = model_names)

    for metric in metrics
        comparison_table[!, string(metric)] = [round(max_val, digits=3) for max_val in metric_maxes[metric]]
    end

    sort_column = comparison_table[!, string(sort_by)]
    comparison_table[!, "Sort_By"] = sort_column

    # Sort the table
    sorted_table = sort(comparison_table, "Sort_By", rev=rev)

    # Remove the sorting column before displaying
    select!(sorted_table, Not("Sort_By"))

    # Display the table
    println("\nComparison of Maximum Metrics Across Models (Sorted by $(string(sort_by))):")
    pretty_table(sorted_table, header=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])

    # return sorted_table
end

function generateComparisonTable(
    model_names::Vector{Any},
    numClasses::Int64,
    metrics::Vector{Symbol},
    metric_maxes_class::Vector{Dict{Symbol, Vector{Any}}}; # Use `;` to define keyword arguments
    sort_by::Symbol = :Accuracy,
    rev::Bool = true
)

    for i in 1:numClasses
        comparison_table = DataFrame(Model = model_names)

        for metric in metrics
            comparison_table[!, string(metric, "_Class_", i)] = [round(max_val, digits=3) for max_val in metric_maxes_class[i][metric]]
        end

        # Add a numeric column for sorting based on the specified metric
        sort_column = comparison_table[!, string(sort_by, "_Class_", i)]
        comparison_table[!, "Sort_By_Class_$(i)"] = sort_column

        # Sort the table
        sorted_table = sort(comparison_table, "Sort_By_Class_$(i)", rev=rev)

        # Remove the sorting column before displaying
        select!(sorted_table, Not("Sort_By_Class_$(i)"))

        # Print the table
        println("\nComparison of Maximum Metrics Across Models for Class $(i) (Sorted by $(string(sort_by))):")
        pretty_table(sorted_table, header=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    end
end


