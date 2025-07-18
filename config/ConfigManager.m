%% ConfigManager.m - 统一配置管理器类 (修复版)
% =========================================================================
% 描述: 集中管理所有仿真初始化参数，提供完整的配置管理功能
% 修复版本：添加缺失的字段以解决运行时错误
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
            % 获取完整的默认配置 - 集中所有初始化参数
            
            % === 1. 基础系统参数 ===
            config.n_stations = 5;
            config.n_components_per_station = [7, 6, 8, 5, 9];
            config.total_components = sum([7, 6, 8, 5, 9]);
            config.total_resources = 100;
            
            % === 2. FSP仿真参数 ===
            config.n_iterations = 100;
            config.n_episodes_per_iter = 50;
            config.max_steps_per_episode = 50; % 添加缺失的字段
            config.max_episode_steps = 50;     % 备用字段名
            config.pool_size_limit = 50;
            config.pool_update_interval = 10;
            config.alpha_ewma = 0.1;  % 策略平均更新参数
            
            % === 3. 强化学习参数 ===
            config.learning_rate = 0.15;
            config.discount_factor = 0.95;
            config.epsilon = 0.4;
            config.epsilon_decay = 0.999;
            config.epsilon_min = 0.05;
            config.temperature = 1.0;
            config.temperature_decay = 0.995;
            config.temperature_min = 0.1;
            
            % === 4. 算法配置 ===
            config.algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
            config.defender_types = config.algorithms; % 兼容性
            config.attacker_algorithm = 'QLearning';
            
            % === 5. 环境参数 ===
            config.state_space_size = sum(config.n_components_per_station) + config.n_stations;
            config.action_space_size = config.total_resources + 1;
            
            % === 6. 攻击模型参数 ===
            config.attack_types = {'malware', 'dos', 'intrusion', 'spoofing', 'tampering', 'eavesdropping'};
            config.attack_severity = [0.8, 0.7, 0.6, 0.5, 0.9, 0.4];
            config.attack_detection_difficulty = [0.6, 0.5, 0.7, 0.8, 0.4, 0.9];
            config.attack_frequency = 0.3;
            config.attack_success_probability = 0.4;
            
            % === 7. 资源模型参数 ===
            config.resource_types = {'computational', 'bandwidth', 'sensors', 'scanning', 'inspection'};
            config.n_resource_types = length(config.resource_types);
            config.resource_effectiveness = [0.7, 0.6, 0.8, 0.5, 0.9];
            config.resource_cost = [1.0, 0.8, 1.2, 0.6, 1.5];
            
            % === 8. RADI (Resource Allocation and Detection Index) 配置 ===
            config.radi = struct();
            config.radi.weight_computation = 0.25;
            config.radi.weight_bandwidth = 0.15;
            config.radi.weight_sensors = 0.25;
            config.radi.weight_scanning = 0.15;
            config.radi.weight_inspection = 0.20;
            config.radi.baseline_detection_rate = 0.7;
            config.radi.optimal_allocation = ones(1, length(config.resource_types)) / length(config.resource_types);
            
            % === 9. 奖励函数参数 ===
            config.reward_params = struct();
            config.reward_params.detection_weight = 1.0;
            config.reward_params.efficiency_weight = 0.5;
            config.reward_params.balance_weight = 0.3;
            config.reward_params.penalty_weight = -0.2;
            config.reward_params.bonus_threshold = 0.8;
            config.reward_params.bonus_multiplier = 1.5;
            
            % === 10. 性能监控参数 ===
            config.performance = struct();
            config.performance.display_interval = 50;
            config.performance.save_interval = 100;
            config.performance.performance_check_interval = 25;
            config.performance.convergence_threshold = 0.01;
            config.performance.convergence_window = 20;
            
            % === 11. 输出配置 ===
            config.output = struct();
            config.output.results_dir = 'results';
            config.output.report_dir = 'reports';
            config.output.models_dir = 'models';
            config.output.checkpoints_dir = 'checkpoints';
            config.output.log_file = fullfile('logs', sprintf('simulation_%s.log', datestr(now, 'yyyymmdd_HHMMSS')));
            config.output.save_models = true;
            config.output.generate_plots = true;
            config.output.export_csv = true;
            
            % === 12. 调试和验证参数 ===
            config.debug = struct();
            config.debug.mode = false;
            config.debug.verbose = false;
            config.debug.plot_realtime = false;
            config.debug.save_states = false;
            config.debug.validation_episodes = 10;
            
            % === 13. 并行计算配置 ===
            config.parallel = struct();
            config.parallel.enabled = false;
            config.parallel.num_workers = 4;
            config.parallel.chunk_size = 10;
            
            % === 14. 高级FSP参数 ===
            config.fsp_advanced = struct();
            config.fsp_advanced.strategy_update_method = 'ewma';  % 'ewma' 或 'uniform'
            config.fsp_advanced.exploration_bonus = 0.1;
            config.fsp_advanced.exploitation_threshold = 0.8;
            config.fsp_advanced.adaptation_rate = 0.05;
            
            % === 15. 网络拓扑参数 ===
            config.network = struct();
            config.network.topology = 'star';  % 'star', 'ring', 'mesh'
            config.network.latency_matrix = ones(config.n_stations) * 0.01;
            config.network.bandwidth_matrix = ones(config.n_stations) * 100;
            
            % === 16. 随机性控制 ===
            config.random_seed = 42;
            
            % === 17. 兼容性设置 ===
            config.compatibility = struct();
            config.compatibility.matlab_version = version('-release');
            config.compatibility.toolbox_required = {'Statistics and Machine Learning Toolbox'};
            
            % === 18. 智能体配置 ===
            config.agents = struct();
            config.agents.defenders = cell(1, numel(config.algorithms));
            for i = 1:numel(config.algorithms)
                config.agents.defenders{i} = struct();
            end
            config.agents.attacker = struct();
            
            % === 19. 确保数组长度一致性 ===
            config = ConfigManager.ensureStationConsistency(config);
        end
        
        function config = ensureStationConsistency(config)
            % 确保所有基于n_stations的配置都是一致的
            
            n_stations = config.n_stations;
            
            % 调整组件数量向量
            if length(config.n_components_per_station) ~= n_stations
                if length(config.n_components_per_station) < n_stations
                    % 扩展数组
                    last_val = config.n_components_per_station(end);
                    config.n_components_per_station = [config.n_components_per_station, ...
                        repmat(last_val, 1, n_stations - length(config.n_components_per_station))];
                else
                    % 截断数组
                    config.n_components_per_station = config.n_components_per_station(1:n_stations);
                end
            end
            
            % 更新总组件数
            config.total_components = sum(config.n_components_per_station);
            
            % 更新状态空间大小
            config.state_space_size = config.total_components + n_stations;
            
            % 调整网络相关矩阵
            if isfield(config, 'network')
                config.network.latency_matrix = ones(n_stations) * 0.01;
                config.network.bandwidth_matrix = ones(n_stations) * 100;
            end
        end
        
        function config = ensureRequiredFields(config)
            % 确保所有必要字段存在，补充缺失字段
            
            default = ConfigManager.getDefaultConfig();
            config = ConfigManager.mergeStructures(default, config);
            
            % 特殊处理：确保episode步数字段存在
            if ~isfield(config, 'max_steps_per_episode') && ~isfield(config, 'max_episode_steps')
                config.max_steps_per_episode = 50;
                config.max_episode_steps = 50;
            elseif isfield(config, 'max_episode_steps') && ~isfield(config, 'max_steps_per_episode')
                config.max_steps_per_episode = config.max_episode_steps;
            elseif isfield(config, 'max_steps_per_episode') && ~isfield(config, 'max_episode_steps')
                config.max_episode_steps = config.max_steps_per_episode;
            end
            
            % 确保数组长度一致性
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
                config = ConfigManager.ensureStationConsistency(config);
            end
            
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
            if length(config.resource_cost) ~= n_resource_types
                config.resource_cost = ConfigManager.adjustArrayLength(config.resource_cost, n_resource_types, 1.0);
            end
            
            % 确保RADI配置完整
            if ~isfield(config, 'radi') || ~isfield(config.radi, 'optimal_allocation')
                config.radi.optimal_allocation = ones(1, n_resource_types) / n_resource_types;
            elseif length(config.radi.optimal_allocation) ~= n_resource_types
                config.radi.optimal_allocation = ones(1, n_resource_types) / n_resource_types;
            end
        end
        
        function adjusted_array = adjustArrayLength(original_array, target_length, default_value)
            % 调整数组长度到目标长度
            
            if length(original_array) == target_length
                adjusted_array = original_array;
            elseif length(original_array) < target_length
                % 扩展数组
                adjusted_array = [original_array, repmat(default_value, 1, target_length - length(original_array))];
            else
                % 截断数组
                adjusted_array = original_array(1:target_length);
            end
        end
        
        function validateConfig(config)
            % 验证配置参数的有效性
            
            % === 基本参数检查 ===
            assert(config.n_stations > 0, '主站数量必须大于0');
            assert(config.n_iterations > 0, '迭代次数必须大于0');
            assert(config.n_episodes_per_iter > 0, '每次迭代的episode数必须大于0');
            assert(config.max_steps_per_episode > 0, '每个episode的最大步数必须大于0');
            assert(config.total_resources > 0, '总资源数必须大于0');
            
            % === 学习参数检查 ===
            assert(config.learning_rate > 0 && config.learning_rate <= 1, '学习率必须在(0,1]范围内');
            assert(config.discount_factor >= 0 && config.discount_factor <= 1, '折扣因子必须在[0,1]范围内');
            assert(config.epsilon >= 0 && config.epsilon <= 1, 'epsilon必须在[0,1]范围内');
            assert(config.epsilon_min >= 0 && config.epsilon_min <= config.epsilon, 'epsilon_min必须在[0,epsilon]范围内');
            
            % === 数组长度检查 ===
            assert(length(config.n_components_per_station) == config.n_stations, ...
                   '组件数量数组长度与主站数量不匹配');
            
            n_attack_types = length(config.attack_types);
            assert(length(config.attack_severity) == n_attack_types, ...
                   '攻击严重程度数组长度与攻击类型数量不匹配');
            assert(length(config.attack_detection_difficulty) == n_attack_types, ...
                   '攻击检测难度数组长度与攻击类型数量不匹配');
            
            n_resource_types = length(config.resource_types);
            assert(length(config.resource_effectiveness) == n_resource_types, ...
                   '资源效率数组长度与资源类型数量不匹配');
            
            % === RADI配置检查 ===
            if isfield(config, 'radi')
                radi_weights = [config.radi.weight_computation, config.radi.weight_bandwidth, ...
                               config.radi.weight_sensors, config.radi.weight_scanning, ...
                               config.radi.weight_inspection];
                assert(abs(sum(radi_weights) - 1.0) < 0.01, 'RADI权重总和应该接近1.0');
            end
            
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
            config_dirs = {};
            if isfield(config, 'output')
                if isfield(config.output, 'results_dir')
                    config_dirs{end+1} = config.output.results_dir;
                end
                if isfield(config.output, 'report_dir')
                    config_dirs{end+1} = config.output.report_dir;
                end
                if isfield(config.output, 'models_dir')
                    config_dirs{end+1} = config.output.models_dir;
                end
                if isfield(config.output, 'checkpoints_dir')
                    config_dirs{end+1} = config.output.checkpoints_dir;
                end
            end
            
            all_dirs = [basic_dirs, config_dirs];
            
            for i = 1:length(all_dirs)
                if ~exist(all_dirs{i}, 'dir')
                    mkdir(all_dirs{i});
                end
            end
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
            config.n_iterations = 20;
            config.n_episodes_per_iter = 10;
            config.max_steps_per_episode = 20;
            config.pool_size_limit = 10;
            
            % 快速收敛参数
            config.learning_rate = 0.3;
            config.epsilon = 0.9;
            config.epsilon_decay = 0.99;
            
            % 简化配置
            config.algorithms = {'QLearning'};
            config.debug.mode = true;
            config.performance.display_interval = 5;
        end
    end
end