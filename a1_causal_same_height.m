clear;
clc;

%% Constants.
dataset_name = "default_15min";
reservoirs_size = 25;

%% Load data.
load(sprintf("data/7_%s_std.mat", dataset_name));
ts = array2table(ts, "VariableNames", col_names);

%% Split by height.
% Assuming cols is a cell array of strings in MATLAB
mass_heights = cellfun(@(x) [strsplit(x, '_'), x], col_names, 'UniformOutput', false);
cols_grouped_height = containers.Map('KeyType','char','ValueType','any');
for i = 1:length(mass_heights)
    mass_height = mass_heights{i};
    height = mass_height{2};
    if isKey(cols_grouped_height, height)
        cols_grouped_height(height) = [cols_grouped_height(height); col_names(i)];
    else
        cols_grouped_height(height) = col_names(i);
    end
end

%% Initialize.
rng(1474);
addpath("functions");
gc_val = struct;

%% Calculate GC statistics.
heights = keys(cols_grouped_height);
for i = 1:length(heights)
    height = heights{i};
    x = ts(:, cols_grouped_height(height)).Vartiables;
    gc_val.(height) = GCx(x, reservoirs_size);
end

%% Export.
save(sprintf("results/a1_%s_gc.mat", dataset_name), "gc_val", "-v7");
