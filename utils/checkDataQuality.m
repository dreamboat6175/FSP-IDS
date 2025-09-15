function [is_valid, issues] = checkDataQuality(results)
%CHECKDATAQUALITY 检查数据质量
% 识别可能导致FSP-TCS:MissingData警告的问题

is_valid = true;
issues = {};

% 检查基本结构
required_fields = {'attacker', 'defenders', 'performance'};
for i = 1:length(required_fields)
    if ~isfield(results, required_fields{i})
        is_valid = false;
        issues{end+1} = sprintf('缺少字段: 