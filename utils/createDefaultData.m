function [attacker, defenders, performance, radi_history] = createDefaultData(n_iterations, n_stations)
%CREATEDEFAULTDATA 创建默认仿真数据结构
% 避免FSP-TCS:MissingData警告

if nargin < 1
    n_iterations = 100;
end
if nargin < 2
    n_stations = 10;
end

fprintf('📝 生成默认数据结构 (迭代: 