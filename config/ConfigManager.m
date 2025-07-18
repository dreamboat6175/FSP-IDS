%% ConfigManager.m - 统一配置管理器类
% =========================================================================
% 描述: 集中管理所有仿真初始化参数，提供完整的配置管理功能
% 版本: v2.0 - 优化版，集中所有参数设置
% =========================================================================

classdef ConfigManager
    
    methods (Static)
        function config = loadConfig(filename)
            % 加载配置文件
            % 输入: filename - 配置文件名（JSON格式）
            % 输出: config - 配置结构体
            
            if nargin < 1 || isempty(filename)
                filename = 'default_config.json';
            end
            
            config_path = fullfile('config', filename);
            
            % 检查配置文件是否存在
            if exist(config_path, 'file')
                try
                    % 读取JSON文件
                    config_text = fileread(config_path);
                    config = jsondecode(config_text);
                    fprintf('✓ 配置文件加载成功: %s\n', filename);
                catch ME
                    warning(ME.identifier, '%s', ME.message);
                    config = ConfigManager.getDefaultConfig();
                end
            else
                fprintf('配置文件不存在，使用默认配置\n');
                config = ConfigManager.getDefaultConfig();
                % 保存默认配置
                ConfigManager.saveConfig(config, 'default_config.json');
            end
            
            % 确保所有必要字段存在
            config = ConfigManager.ensureRequiredFields(config);
            
            % 验证配置有效性
            ConfigManager.validateConfig(config);
        end
        
        function config = getDefaultConfig()
            % 获取完整的默认配置 - 确保所有智能体参数都被正确设置
            
            % === 1. 基础系统参数 ===
            config.n_stations = 5;
            config.n_components_per_station = [7, 6, 8, 5, 9];
            config.total_components = sum([7, 6, 8, 5, 9]);
            config.total_resources = 100;
            
            % === 2. FSP仿真参数 ===
            config.n_iterations = 1000;
            config.n_episodes_per_iter = 100;
            config.pool_size_limit = 50;
            config.pool_update_interval = 10;
            config.alpha_ewma = 0.1;
            
            % === 3. 强化学习参数（主要配置） ===
            config.learning_rate = 0.15;
            config.discount_factor = 0.95;
            config.epsilon = 0.4;
            config.epsilon_decay = 0.999;
            config.epsilon_min = 0.05;
            config.temperature = 1.0;
            config.temperature_decay = 0.995;
            
            % === 4. 算法配置 ===
            config.algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
            config.defender_types = config.algorithms;
            
            % === 5. 攻击系统配置 ===
            config.attack_types = {'wireless_spoofing', 'dos_attack', 'semantic_attack', ...
                                  'supply_chain', 'protocol_exploit', 'maintenance_port'};
            config.attack_severity = [0.3, 0.5, 0.7, 0.4, 0.6, 0.8];
            config.attack_detection_difficulty = [0.4, 0.3, 0.7, 0.8, 0.6, 0.5];
            
            % === 6. 资源系统配置 ===
            config.resource_types = {'computation', 'bandwidth', 'sensors', ...
                                   'scanning_freq', 'inspection_depth'};
            config.resource_effectiveness = [0.7, 0.6, 0.8, 0.5, 0.9];
            
            % === 7. 检测系统参数 ===
            config.detection_enabled = true;
            config.base_detection_rate = 0.3;
            config.detection_sensitivity = 0.8;
            config.false_positive_rate = 0.1;
            
            % === 8. RADI评估体系配置 ===
            config.radi = struct();
            config.radi.optimal_allocation = ones(1, config.n_stations) / config.n_stations;
            config.radi.weight_computation = 0.3;
            config.radi.weight_bandwidth = 0.2;
            config.radi.weight_sensors = 0.2;
            config.radi.weight_scanning = 0.15;
            config.radi.weight_inspection = 0.15;
            config.radi.threshold_excellent = 0.1;
            config.radi.threshold_good = 0.2;
            config.radi.threshold_acceptable = 0.3;
            
            % === 9. 奖励函数配置 ===
            config.reward = struct();
            config.reward.w_radi = 0.4;
            config.reward.w_efficiency = 0.3;
            config.reward.w_balance = 0.3;
            config.reward.w_class = 0.5;
            config.reward.w_cost = 0.1;
            config.reward.w_process = 0.4;
            config.reward.true_positive = 100;
            config.reward.true_negative = 20;
            config.reward.false_positive = -3;
            config.reward.false_negative = -150;
            
            % === 10. 性能监控配置 ===
            config.performance = struct();
            config.performance.display_interval = 50;
            config.performance.save_interval = 100;
            config.performance.param_update_interval = 50;
            config.performance.convergence_threshold = 0.01;
            
            % === 11. 输出配置 ===
            config.output = struct();
            config.output.visualization = true;
            config.output.generate_report = true;
            config.output.save_models = false;
            config.output.save_checkpoints = false;
            config.output.checkpoint_interval = 500;
            config.output.results_dir = 'results';
            config.output.report_dir = 'reports';
            
            % 时间戳
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            config.output.log_file = sprintf('logs/simulation_%s.log', timestamp);
            
            % === 12. 随机种子 ===
            config.random_seed = 42;
            
            % ===== 关键修复：智能体配置 =====
            % 确保每个智能体都能获得完整的配置参数
            config.agents = struct();
            
            % 防御者智能体配置
            config.agents.defenders = cell(1, length(config.algorithms));
            for i = 1:length(config.algorithms)
                config.agents.defenders{i} = struct();
                config.agents.defenders{i}.learning_rate = config.learning_rate;
                config.agents.defenders{i}.discount_factor = config.discount_factor;
                config.agents.defenders{i}.epsilon = config.epsilon;
                config.agents.defenders{i}.epsilon_decay = config.epsilon_decay;
                config.agents.defenders{i}.epsilon_min = config.epsilon_min;
                config.agents.defenders{i}.temperature = config.temperature;
                config.agents.defenders{i}.temperature_decay = config.temperature_decay;
                config.agents.defenders{i}.n_stations = config.n_stations;
                config.agents.defenders{i}.total_resources = config.total_resources;
            end
            
            % 攻击者智能体配置
            config.agents.attacker = struct();
            config.agents.attacker.learning_rate = config.learning_rate;
            config.agents.attacker.discount_factor = config.discount_factor;
            config.agents.attacker.epsilon = config.epsilon;
            config.agents.attacker.epsilon_decay = config.epsilon_decay;
            config.agents.attacker.epsilon_min = config.epsilon_min;
            config.agents.attacker.temperature = config.temperature;
            config.agents.attacker.temperature_decay = config.temperature_decay;
            config.agents.attacker.n_stations = config.n_stations;
            
            % === 13. 确保所有基于n_stations的配置都正确 ===
            config = ensureStationConsistency(config);
        end
        
        % === 添加新的辅助函数：确保站点相关配置的一致性 ===
        function config = ensureStationConsistency(config)
            % 确保所有基于n_stations的配置都是一致的
            
            n_stations = config.n_stations;
            
            % 调整组件数量向量
            if length(config.n_components_per_station) ~= n_stations
                if length(config.n_components_per_station) < n_stations
                    % 扩展：重复最后一个值
                    last_val = config.n_components_per_station(end);
                    config.n_components_per_station = [config.n_components_per_station, ...
                                                      repmat(last_val, 1, n_stations - length(config.n_components_per_station))];
                else
                    % 截断
                    config.n_components_per_station = config.n_components_per_station(1:n_stations);
                end
                config.total_components = sum(config.n_components_per_station);
            end
            
            % 调整攻击检测难度
            if length(config.attack_detection_difficulty) ~= length(config.attack_types)
                config.attack_detection_difficulty = ConfigManager.adjustArrayLength(config.attack_detection_difficulty, length(config.attack_types), 0.5);
            end
            
            % 调整攻击严重程度
            if length(config.attack_severity) ~= length(config.attack_types)
                config.attack_severity = ConfigManager.adjustArrayLength(config.attack_severity, length(config.attack_types), 0.5);
            end
            
            % 调整资源效果
            if length(config.resource_effectiveness) ~= length(config.resource_types)
                config.resource_effectiveness = ConfigManager.adjustArrayLength(config.resource_effectiveness, length(config.resource_types), 0.7);
            end
            

            function adjusted_array = adjustArrayLength(array, target_length, default_value)
                % 调整数组长度到目标长度
                if length(array) < target_length
                    % 扩展数组
                    adjusted_array = [array, repmat(default_value, 1, target_length - length(array))];
                elseif length(array) > target_length
                    % 截断数组
                    adjusted_array = array(1:target_length);
                else
                    adjusted_array = array;
                end
            end    

            % 调整RADI最优分配
            config.radi.optimal_allocation = ones(1, n_stations) / n_stations;
            
            % 更新所有智能体配置中的n_stations
            if isfield(config, 'agents')
                if isfield(config.agents, 'defenders')
                    for i = 1:length(config.agents.defenders)
                        config.agents.defenders{i}.n_stations = n_stations;
                    end
                end
                if isfield(config.agents, 'attacker')
                    config.agents.attacker.n_stations = n_stations;
                end
            end
            
            fprintf('[ConfigManager] 已调整配置以匹配 %d 个站点\n', n_stations);
        end
        
        function adjusted_array = adjustArrayLength(array, target_length, default_value)
            % 调整数组长度到目标长度
            if length(array) < target_length
                % 扩展数组
                adjusted_array = [array, repmat(default_value, 1, target_length - length(array))];
            elseif length(array) > target_length
                % 截断数组
                adjusted_array = array(1:target_length);
            else
                adjusted_array = array;
            end
        end    
        
        function config = getOptimizedConfig()
            % 获取性能优化的配置
            
            config = ConfigManager.getDefaultConfig();
            
            % 优化学习参数以提高收敛速度
            config.learning_rate = 0.25;
            config.epsilon = 0.8;
            config.epsilon_decay = 0.9995;
            config.epsilon_min = 0.15;
            config.temperature = 2.0;
            
            % 降低攻击检测难度以验证算法效果
            config.attack_detection_difficulty = [0.2, 0.15, 0.3, 0.4, 0.25, 0.35];
            
            % 提高资源效率
            config.resource_effectiveness = [0.8, 0.7, 0.85, 0.6, 0.95];
            
            % 增加迭代数和总资源
            config.n_iterations = 1500;
            config.total_resources = 150;
            
            % 更频繁的性能检查
            config.performance.display_interval = 20;
            config.performance.performance_check_interval = 10;
        end
        
        function config = getTestConfig()
            % 获取测试用的小规模快速配置
            
            config = ConfigManager.getDefaultConfig();
            
            % 小规模参数
            config.n_stations = 3;
            config.n_components_per_station = [3, 3, 3];
            config.total_components = 9;
            config.total_resources = 50;
            
            % 快速测试参数
            config.n_iterations = 50;
            config.n_episodes_per_iter = 20;
            config.pool_size_limit = 10;
            
            % 快速收敛参数
            config.learning_rate = 0.3;
            config.epsilon = 0.9;
            config.epsilon_decay = 0.99;
            
            % 简化配置
            config.algorithms = {'QLearning'};
            config.debug.mode = true;
            config.performance.display_interval = 10;
        end
        
        function config = ensureRequiredFields(config)
            % 确保所有必要字段存在，补充缺失字段
            
            default = ConfigManager.getDefaultConfig();
            config = ConfigManager.mergeStructures(default, config);
            
            % 特殊处理：确保数组长度一致性
            config = ConfigManager.validateArrayLengths(config);
        end
        
        function config = mergeStructures(default, user_config)
            % 递归合并结构体，保留用户设置，补充默认值
            
            config = default;
            
            if ~isstruct(user_config)
                return;
            end
            
            fields = fieldnames(user_config);
            for i = 1:length(fields)
                field = fields{i};
                if isfield(default, field)
                    if isstruct(default.(field)) && isstruct(user_config.(field))
                        % 递归合并子结构
                        config.(field) = ConfigManager.mergeStructures(default.(field), user_config.(field));
                    else
                        % 直接使用用户配置
                        config.(field) = user_config.(field);
                    end
                else
                    % 用户新增字段
                    config.(field) = user_config.(field);
                end
            end
        end
        
        function config = validateArrayLengths(config)
            % 验证并修正数组长度一致性
            
            % 修正组件数量数组
            if length(config.n_components_per_station) ~= config.n_stations
                if length(config.n_components_per_station) < config.n_stations
                    % 扩展到正确长度
                    config.n_components_per_station = [config.n_components_per_station, ...
                        repmat(config.n_components_per_station(end), ...
                        1, config.n_stations - length(config.n_components_per_station))];
                else
                    % 截断到正确长度
                    config.n_components_per_station = config.n_components_per_station(1:config.n_stations);
                end
            end
            
            % 更新总组件数
            config.total_components = sum(config.n_components_per_station);
            
            % 修正攻击相关数组
            n_attack_types = length(config.attack_types);
            if length(config.attack_severity) ~= n_attack_types
                config.attack_severity = ConfigManager.adjustArrayLength(config.attack_severity, n_attack_types, 0.5);
            end
            if length(config.attack_detection_difficulty) ~= n_attack_types
                config.attack_detection_difficulty = ConfigManager.adjustArrayLength(config.attack_detection_difficulty, n_attack_types, 0.5);
            end
            
            % 修正资源相关数组
            n_resource_types = length(config.resource_types);
            if length(config.resource_effectiveness) ~= n_resource_types
                config.resource_effectiveness = ConfigManager.adjustArrayLength(config.resource_effectiveness, n_resource_types, 0.7);
            end
            if length(config.radi.optimal_allocation) ~= n_resource_types
                config.radi.optimal_allocation = ones(1, n_resource_types) / n_resource_types;
            end
        end
        
        function validateConfig(config)
            % 全面验证配置参数的合法性
            
            fprintf('正在验证配置参数...\n');
            
            % === 基本参数检查 ===
            assert(config.n_stations > 0, 'n_stations必须大于0');
            assert(length(config.n_components_per_station) == config.n_stations, ...
                   '组件数量数组长度必须等于站点数量');
            assert(config.total_resources > 0, 'total_resources必须大于0');
            
            % === 学习参数检查 ===
            assert(config.learning_rate > 0 && config.learning_rate <= 1, ...
                   'learning_rate必须在(0,1]范围内');
            assert(config.discount_factor >= 0 && config.discount_factor <= 1, ...
                   'discount_factor必须在[0,1]范围内');
            assert(config.epsilon >= 0 && config.epsilon <= 1, ...
                   'epsilon必须在[0,1]范围内');
            assert(config.epsilon_decay >= 0 && config.epsilon_decay <= 1, ...
                   'epsilon_decay必须在[0,1]范围内');
            
            % === 仿真参数检查 ===
            assert(config.n_iterations > 0, 'n_iterations必须大于0');
            assert(config.n_episodes_per_iter > 0, 'n_episodes_per_iter必须大于0');
            
            % === 数组长度一致性检查 ===
            n_attack_types = length(config.attack_types);
            assert(length(config.attack_severity) == n_attack_types, ...
                   '攻击严重程度数组长度与攻击类型数量不匹配');
            assert(length(config.attack_detection_difficulty) == n_attack_types, ...
                   '检测难度数组长度与攻击类型数量不匹配');
            
            n_resource_types = length(config.resource_types);
            assert(length(config.resource_effectiveness) == n_resource_types, ...
                   '资源效率数组长度与资源类型数量不匹配');
            
            % === RADI配置检查 ===
            radi_weights = [config.radi.weight_computation, config.radi.weight_bandwidth, ...
                           config.radi.weight_sensors, config.radi.weight_scanning, ...
                           config.radi.weight_inspection];
            assert(abs(sum(radi_weights) - 1.0) < 0.01, 'RADI权重总和应该接近1.0');
            
            % === 创建必要目录 ===
            ConfigManager.createDirectories(config);
            
            fprintf('✓ 配置验证通过\n');
        end
        
        function saveConfig(config, filename)
            % 保存配置到JSON文件
            
            if nargin < 2
                filename = sprintf('config_backup_%s.json', datestr(now, 'yyyymmdd_HHMMSS'));
            end
            
            config_path = fullfile('config', filename);
            
            try
                % 确保目录存在
                if ~exist('config', 'dir')
                    mkdir('config');
                end
                
                % 保存为格式化JSON
                config_json = jsonencode(config, 'PrettyPrint', true);
                fid = fopen(config_path, 'w');
                fprintf(fid, '%s', config_json);
                fclose(fid);
                
                fprintf('✓ 配置已保存: %s\n', config_path);
            catch ME
                warning('配置保存失败: %s', ME.message);
            end
        end
        
        function createDirectories(config)
            % 创建所有必要的目录结构
            
            % 基础目录
            basic_dirs = {'logs', 'results', 'reports', 'config', 'data'};
            
            % 从配置中获取的目录
            config_dirs = {config.output.results_dir, config.output.report_dir, ...
                          config.output.models_dir, config.output.checkpoints_dir};
            
            all_dirs = [basic_dirs, config_dirs];
            
            for i = 1:length(all_dirs)
                if ~isempty(all_dirs{i}) && ~exist(all_dirs{i}, 'dir')
                    mkdir(all_dirs{i});
                end
            end
        end
        
        function config = updateLearningParameters(config, iteration)
            % 动态更新学习参数（支持自适应学习）
            
            % 更新探索率
            config.epsilon = max(config.epsilon_min, ...
                               config.epsilon * config.epsilon_decay);
            
            % 可选：更新学习率（渐减策略）
            if iteration > 100
                decay_factor = 1 / (1 + 0.001 * (iteration - 100));
                config.learning_rate = max(0.01, config.learning_rate * decay_factor);
            end
            
            % 更新温度参数
            if isfield(config, 'temperature')
                config.temperature = max(0.1, config.temperature * config.temperature_decay);
            end
        end
        
        function displayConfigSummary(config)
            % 显示配置摘要信息
            
            fprintf('\n=== FSP-TCS 配置摘要 ===\n');
            fprintf('系统配置:\n');
            fprintf('  站点数量: %d\n', config.n_stations);
            fprintf('  总组件数: %d\n', config.total_components);
            fprintf('  总资源数: %d\n', config.total_resources);
            fprintf('\n学习配置:\n');
            fprintf('  学习率: %.3f\n', config.learning_rate);
            fprintf('  探索率: %.3f (最小: %.3f)\n', config.epsilon, config.epsilon_min);
            fprintf('  折扣因子: %.3f\n', config.discount_factor);
            fprintf('\n仿真配置:\n');
            fprintf('  迭代次数: %d\n', config.n_iterations);
            fprintf('  每轮episodes: %d\n', config.n_episodes_per_iter);
            fprintf('  算法: %s\n', strjoin(config.algorithms, ', '));
            fprintf('\n输出配置:\n');
            fprintf('  可视化: %s\n', config.output.visualization);
            fprintf('  生成报告: %s\n', config.output.generate_report);
            fprintf('  结果目录: %s\n', config.output.results_dir);
            fprintf('========================\n\n');
        end
    end
end