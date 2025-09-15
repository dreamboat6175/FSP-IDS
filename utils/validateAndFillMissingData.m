function validated_results = validateAndFillMissingData(results, config)
%VALIDATEANDFILLMISSINGDATA 验证并填充缺失的数据
% 避免FSP-TCS:MissingData警告

validated_results = results;

% 获取基本配置
n_iterations = getConfigValue(config, 'simulation.n_iterations', 100);
n_stations = getConfigValue(config, 'system.n_stations', 10);

% 验证和填充攻击者数据
if ~isfield(validated_results, 'attacker') || isempty(validated_results.attacker)
    fprintf('⚠️ 警告: 攻击者数据缺失，使用默认结构\n');
    validated_results.attacker = createDefaultAttackerData(n_iterations, n_stations);
else
    validated_results.attacker = validateAttackerData(validated_results.attacker, n_iterations, n_stations);
end

% 验证和填充防御者数据
if ~isfield(validated_results, 'defenders') || isempty(validated_results.defenders)
    fprintf('⚠️ 警告: 防御者数据缺失，使用默认结构\n');
    validated_results.defenders = createDefaultDefendersData(n_iterations);
else
    validated_results.defenders = validateDefendersData(validated_results.defenders, n_iterations);
end

% 验证和填充性能指标
if ~isfield(validated_results, 'performance') || isempty(validated_results.performance)
    fprintf('⚠️ 警告: 性能数据缺失，使用默认值\n');
    validated_results.performance = createDefaultPerformanceData(n_iterations);
else
    validated_results.performance = validatePerformanceData(validated_results.performance, n_iterations);
end

% 验证RADI历史数据
if ~isfield(validated_results, 'radi_history') || isempty(validated_results.radi_history)
    fprintf('⚠️ 警告: RADI历史数据缺失，使用默认趋势\n');
    validated_results.radi_history = createDefaultRadiHistory(n_iterations);
end

end

function value = getConfigValue(config, field_path, default_value)
%获取配置值，支持嵌套字段
try
    fields = strsplit(field_path, '.');
    value = config;
    for i = 1:length(fields)
        if isfield(value, fields{i})
            value = value.(fields{i});
        else
            value = default_value;
            return;
        end
    end
catch
    value = default_value;
end
end
