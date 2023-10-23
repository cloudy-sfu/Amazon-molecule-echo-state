clear;
clc;

%% Constants.
dataset_name = "default_15min";
reservoirs_size = 25;

%% Load data.
load(sprintf("data/7_%s_std.mat", dataset_name));
ts = array2table(ts, "VariableNames", col_names);

%% Split by mass.
% Assuming cols is a cell array of strings in MATLAB
mass_heights = cellfun(@(x) [strsplit(x, '_'), x], col_names, 'UniformOutput', false);
cols_grouped_mass = containers.Map('KeyType','char','ValueType','any');
all_heights = cell(0, 1);
for i = 1:length(mass_heights)
    mass_height = mass_heights{i};
    mass = mass_height{1};
    all_heights = [all_heights; mass_height{2}];
    if isKey(cols_grouped_mass, mass)
        cols_grouped_mass(mass) = [cols_grouped_mass(mass); col_names(i)];
    else
        cols_grouped_mass(mass) = col_names(i);
    end
end
all_heights = unique(all_heights);
m = size(all_heights, 1);

%% Initialize.
rng(1474);
addpath("functions");
gc_val = struct;

%% Calculate GC statistics.
masses = keys(cols_grouped_mass);
for i = 1:length(masses)
    mass = masses{i};

    col_names = cols_grouped_mass(mass);
    n = size(col_names, 1);
    h_idx = zeros(n, 1);
    for k = 1:n
        mass_height = strsplit(col_names{k}, '_');
        height = mass_height{2};
        h_idx(k) = find(strcmp(all_heights, height));
    end

    x = ts(:, cols_grouped_mass(mass)).Variables;
    gc_val.(mass) = zeros(m);
    gc_val.(mass)(h_idx, h_idx) = GCx(x, reservoirs_size);
end

%% Export.
save(sprintf("results/a2_%s_gc.mat", dataset_name), "gc_val", "all_heights", "-v7");
