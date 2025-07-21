%% ConfigManager.m - 统一配置管理器类 (改进版)
% =========================================================================
% 描述: 集中管理所有仿真初始化参数，提供完整的配置管理功能
% 改进版本：实现了结构优化、参数合理性改进和代码逻辑增强
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
            
            % 处理日志文件名（只在未定义时生成）
            if ~isfield(config.output, 'log_file') || isempty(config.output.log_file)
                log_dir = 'logs';
                if ~exist(log_dir, 'dir'), mkdir(log_dir); end
                config.output.log_file = fullfile(log_dir, sprintf('simulation_%s.log', datestr(now, 'yyyymmdd_HHMMSS')));
            end
            
            % 验证配置有效性
            ConfigManager.validateConfig(config);
        end
        
        function config = getDefaultConfig()
            % 获取完整的默认配置 - 集中所有初始化参数
            
            % === 1. 基础系统参数 ===
            config.n_stations = 10;
            config.n_components_per_station = [7, 6, 8, 5, 9, 15, 4, 6, 3, 4];
            config.total_components = sum(config.n_components_per_station);
            config.total_resources = 100;
            
            % === 2. FSP仿真参数 ===
            config.n_iterations = 1000;
            config.n_episodes_per_iter = 50;
            config.max_episode_steps = 50;     % 统一使用这个字段名
            config.pool_size_limit = 50;
            config.pool_update_interval = 10;
            config.alpha_ewma = 0.1;  % 策略平均更新参数
            
            % === 3. 强化学习默认参数 ===
            config.rl_defaults = struct();
            config.rl_defaults.learning_rate = 0.15;
            config.rl_defaults.discount_factor = 0.95;
            
            % 探索策略配置
            config.rl_defaults.exploration_strategy = 'epsilon-greedy'; % 可选: 'softmax', 'none'
            
            % Epsilon-Greedy 参数
            config.rl_defaults.epsilon_greedy = struct(...
                'epsilon', 0.4, ...
                'epsilon_decay', 0.999, ...
                'epsilon_min', 0.05 ...
            );
            
            % Softmax (Boltzmann) 参数
            config.rl_defaults.softmax_exploration = struct(...
                'temperature', 1.0, ...
                'temperature_decay', 0.995, ...
                'temperature_min', 0.1 ...
            );
            
            % === 4. 算法配置 ===
            config.algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
            config.attacker_algorithm = 'QLearning';
            
            % === 5. 环境参数 ===
            config.state_space_size = sum(config.n_components_per_station) + config.n_stations;
            config.action_space_size = config.total_resources + 1;
            
            % === 6. 攻击模型参数 (改进后的结构体数组) ===
            config.attacks = [ ...
                struct('name', 'malware',       'severity', 0.8, 'detection_difficulty', 0.6), ...
                struct('name', 'dos',           'severity', 0.7, 'detection_difficulty', 0.5), ...
                struct('name', 'intrusion',     'severity', 0.6, 'detection_difficulty', 0.7), ...
                struct('name', 'spoofing',      'severity', 0.5, 'detection_difficulty', 0.8), ...
                struct('name', 'tampering',     'severity', 0.9, 'detection_difficulty', 0.4), ...
                struct('name', 'eavesdropping', 'severity', 0.4, 'detection_difficulty', 0.9)  ...
            ];
            config.attack_frequency = 0.3;
            config.attack_success_probability = 0.4;
            
            % 兼容性支持（旧字段名）
            config.attack_types = {config.attacks.name};
            config.attack_severity = [config.attacks.severity];
            config.attack_detection_difficulty = [config.attacks.detection_difficulty];
            
            % === 7. 资源模型参数 (改进后的结构体数组) ===
            config.resources = [ ...
                struct('name', 'computational', 'effectiveness', 0.7, 'cost', 1.0), ...
                struct('name', 'bandwidth',     'effectiveness', 0.6, 'cost', 0.8), ...
                struct('name', 'sensors',       'effectiveness', 0.8, 'cost', 1.2), ...
                struct('name', 'scanning',      'effectiveness', 0.5, 'cost', 0.6), ...
                struct('name', 'inspection',    'effectiveness', 0.9, 'cost', 1.5)  ...
            ];
            config.n_resource_types = length(config.resources);
            
            % 兼容性支持（旧字段名）
            config.resource_types = {config.resources.name};
            config.resource_effectiveness = [config.resources.effectiveness];
            config.resource_cost = [config.resources.cost];
            
            % === 8. RADI (Resource Allocation and Detection Index) 配置 ===
            config.radi = struct();
            % 使用结构体存储权重，更灵活
            config.radi.weights = struct(...
                'computational', 0.25, ...
                'bandwidth',     0.15, ...
                'sensors',       0.25, ...
                'scanning',      0.15, ...
                'inspection',    0.20  ...
            );
            
            % 兼容性支持（旧字段名）
            config.radi.weight_computation = config.radi.weights.computational;
            config.radi.weight_bandwidth = config.radi.weights.bandwidth;
            config.radi.weight_sensors = config.radi.weights.sensors;
            config.radi.weight_scanning = config.radi.weights.scanning;
            config.radi.weight_inspection = config.radi.weights.inspection;
            
            config.radi.baseline_detection_rate = 0.7;
            config.radi.optimal_allocation = ones(1, length(config.resources)) / length(config.resources);
            
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
            % 注意：log_file 在 loadConfig 中生成，避免每次调用都生成新的文件名
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
            
            % === 13. 并行计算配置（动态配置） ===
            config.parallel = struct();
            config.parallel.enabled = false;
            % 动态获取工作进程数
            try
                % 尝试获取本地集群的默认工作进程数
                p = gcp('nocreate');
                if isempty(p)
                    myCluster = parcluster('local');
                    config.parallel.num_workers = myCluster.NumWorkers;
                else
                    config.parallel.num_workers = p.NumWorkers;
                end
            catch
                % 如果出错，回退到安全值
                config.parallel.num_workers = 4;
            end
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
            
            % === 18. 智能体配置（支持异构智能体） ===
            config.agents = struct();
            config.agents.defenders = cell(1, numel(config.algorithms));
            for i = 1:numel(config.algorithms)
                config.agents.defenders{i} = struct();
                config.agents.defenders{i}.algorithm = config.algorithms{i};
                config.agents.defenders{i}.name = sprintf('Defender_%s', config.algorithms{i});
                
                % 可以为特定算法覆盖默认值
                if strcmp(config.algorithms{i}, 'QLearning')
                    config.agents.defenders{i}.learning_rate = 0.2; % 为QLearning设置不同的学习率
                elseif strcmp(config.algorithms{i}, 'SARSA')
                    config.agents.defenders{i}.epsilon = 0.5; % 为SARSA设置更高的探索率
                end
            end
            config.agents.attacker = struct('algorithm', 'QLearning', 'name', 'Attacker');
            
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
            
            % 特殊处理：优先使用 max_episode_steps
            if isfield(config, 'max_steps_per_episode') && ~isfield(config, 'max_episode_steps')
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
                    % =======================【代码修复 v2】=======================
                    % 检查字段是否为【标量】结构体 (scalar struct), 而非结构体数组.
                    % 对结构体数组 (如 config.attacks 或从 JSON 加载的 config.agents.defenders)
                    % 的递归会导致 "输入参数过多" 的错误, 因为 default.(field) 会返回一个逗号分隔的列表.
                    % 对于结构体数组或普通数组, 我们直接采用用户的配置, 不进行递归合并.
                    % ===========================================================
                    isDefaultScalarStruct = isstruct(default.(field)) && isscalar(default.(field));
                    isUserScalarStruct = isstruct(user_config.(field)) && isscalar(user_config.(field));

                    if isDefaultScalarStruct && isUserScalarStruct
                        % 仅当双方都是标量结构体时, 才进行递归合并
                        config.(field) = ConfigManager.mergeStructures(default.(field), user_config.(field));
                    else
                        % 否则, 直接使用用户配置覆盖默认值
                        % 这会正确处理结构体数组、普通数组和基本类型
                        config.(field) = user_config.(field);
                    end
                else
                    % 用户新增的字段, 直接添加
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
            
            % 修正攻击相关数组（如果使用旧格式）
            if isfield(config, 'attack_types') && ~isfield(config, 'attacks')
                n_attack_types = length(config.attack_types);
                if length(config.attack_severity) ~= n_attack_types
                    config.attack_severity = ConfigManager.adjustArrayLength(config.attack_severity, n_attack_types, 0.5);
                end
                if length(config.attack_detection_difficulty) ~= n_attack_types
                    config.attack_detection_difficulty = ConfigManager.adjustArrayLength(config.attack_detection_difficulty, n_attack_types, 0.5);
                end
            end
            
            % 修正资源相关数组（如果使用旧格式）
            if isfield(config, 'resource_types') && ~isfield(config, 'resources')
                n_resource_types = length(config.resource_types);
                if length(config.resource_effectiveness) ~= n_resource_types
                    config.resource_effectiveness = ConfigManager.adjustArrayLength(config.resource_effectiveness, n_resource_types, 0.7);
                end
                if length(config.resource_cost) ~= n_resource_types
                    config.resource_cost = ConfigManager.adjustArrayLength(config.resource_cost, n_resource_types, 1.0);
                end
            end
            
            % 确保RADI配置完整
            if isfield(config, 'resources')
                n_resource_types = length(config.resources);
            else
                n_resource_types = length(config.resource_types);
            end
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
            assert(config.max_episode_steps > 0, '每个episode的最大步数必须大于0');
            assert(config.total_resources > 0, '总资源数必须大于0');
            
            % === 学习参数检查 ===
            rl_defaults = config.rl_defaults;
            assert(rl_defaults.learning_rate > 0 && rl_defaults.learning_rate <= 1, '学习率必须在(0,1]范围内');
            assert(rl_defaults.discount_factor >= 0 && rl_defaults.discount_factor <= 1, '折扣因子必须在[0,1]范围内');
            
            % 根据探索策略检查相应参数
            if strcmp(rl_defaults.exploration_strategy, 'epsilon-greedy')
                eps_params = rl_defaults.epsilon_greedy;
                assert(eps_params.epsilon >= 0 && eps_params.epsilon <= 1, 'epsilon必须在[0,1]范围内');
                assert(eps_params.epsilon_min >= 0 && eps_params.epsilon_min <= eps_params.epsilon, ...
                    'epsilon_min必须在[0,epsilon]范围内');
            elseif strcmp(rl_defaults.exploration_strategy, 'softmax')
                temp_params = rl_defaults.softmax_exploration;
                assert(temp_params.temperature > 0, 'temperature必须大于0');
                assert(temp_params.temperature_min > 0, 'temperature_min必须大于0');
            end
            
            % === 数组长度检查 ===
            assert(length(config.n_components_per_station) == config.n_stations, ...
                   '组件数量数组长度与主站数量不匹配');
            
            % 检查攻击和资源数组（支持新旧格式）
            if isfield(config, 'attacks')
                assert(length(config.attacks) > 0, '攻击类型数组不能为空');
            else
                n_attack_types = length(config.attack_types);
                assert(length(config.attack_severity) == n_attack_types, ...
                       '攻击严重程度数组长度与攻击类型数量不匹配');
                assert(length(config.attack_detection_difficulty) == n_attack_types, ...
                       '攻击检测难度数组长度与攻击类型数量不匹配');
            end
            
            if isfield(config, 'resources')
                assert(length(config.resources) > 0, '资源类型数组不能为空');
            else
                n_resource_types = length(config.resource_types);
                assert(length(config.resource_effectiveness) == n_resource_types, ...
                       '资源效率数组长度与资源类型数量不匹配');
            end
            
            % === RADI配置检查 ===
            if isfield(config, 'radi') && isfield(config.radi, 'weights')
                % 新格式：检查权重结构体
                weights = struct2array(config.radi.weights);
                assert(abs(sum(weights) - 1.0) < 0.01, 'RADI权重总和应该接近1.0');
            elseif isfield(config, 'radi')
                % 旧格式：检查独立权重字段
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
        end
        
        function createDirectories(config)
            % 创建所有必要的目录结构
            
            % 基础目录
            basic_dirs = {'logs', 'results', 'reports', 'config', 'data', 'models', 'checkpoints'};
            
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
            config.max_episode_steps = 20;
            config.pool_size_limit = 10;
            
            % 快速收敛参数
            config.rl_defaults.learning_rate = 0.3;
            config.rl_defaults.epsilon_greedy.epsilon = 0.9;
            config.rl_defaults.epsilon_greedy.epsilon_decay = 0.99;
            
            % 简化配置
            config.algorithms = {'QLearning'};
            config.debug.mode = true;
            config.performance.display_interval = 5;
            
            % 重新确保一致性
            config = ConfigManager.ensureStationConsistency(config);
        end
        
        function mergeAgentConfig(agent, config)
            % 合并智能体特定配置与默认配置
            % 输入: agent - 智能体对象
            %       config - 全局配置
            
            % 首先应用默认RL参数
            if isfield(config, 'rl_defaults')
                % 基本参数
                if isfield(config.rl_defaults, 'learning_rate')
                    agent.learning_rate = config.rl_defaults.learning_rate;
                end
                if isfield(config.rl_defaults, 'discount_factor')
                    agent.discount_factor = config.rl_defaults.discount_factor;
                end
                
                % 探索策略参数
                if isfield(config.rl_defaults, 'exploration_strategy')
                    agent.exploration_strategy = config.rl_defaults.exploration_strategy;
                    
                    % 根据策略类型设置相应参数
                    if strcmp(agent.exploration_strategy, 'epsilon-greedy') && ...
                       isfield(config.rl_defaults, 'epsilon_greedy')
                        eps_params = config.rl_defaults.epsilon_greedy;
                        agent.epsilon = eps_params.epsilon;
                        agent.epsilon_decay = eps_params.epsilon_decay;
                        agent.epsilon_min = eps_params.epsilon_min;
                    elseif strcmp(agent.exploration_strategy, 'softmax') && ...
                           isfield(config.rl_defaults, 'softmax_exploration')
                        temp_params = config.rl_defaults.softmax_exploration;
                        agent.temperature = temp_params.temperature;
                        agent.temperature_decay = temp_params.temperature_decay;
                        agent.temperature_min = temp_params.temperature_min;
                    end
                end
            end
            
            % 然后应用智能体特定配置（如果存在）
            if isfield(config, 'agents')
                % 查找对应的智能体配置
                agent_config = [];
                if strcmp(agent.agent_type, 'defender') && isfield(config.agents, 'defenders')
                    % =======================【代码修复 v1】=======================
                    % 修复1: 从JSON加载配置时, defenders 是结构体数组, 必须用圆括号 () 索引。
                    % 修复2: 智能体名称(agent.name)如 'defender_QLearning_1',
                    %        配置中算法名称(algorithm)为 'QLearning'。
                    %        使用 'contains' 进行模糊匹配, 而不是 'strcmp'。
                    % =========================================================
                    for i = 1:length(config.agents.defenders)
                        % 使用圆括号 (i) 索引结构体数组
                        if isfield(config.agents.defenders(i), 'algorithm') && ...
                           contains(agent.name, config.agents.defenders(i).algorithm, 'IgnoreCase', true)
                            agent_config = config.agents.defenders(i);
                            break;
                        end
                    end
                elseif strcmp(agent.agent_type, 'attacker') && isfield(config.agents, 'attacker')
                    agent_config = config.agents.attacker;
                end
                
                % 应用找到的特定配置
                if ~isempty(agent_config)
                    fields = fieldnames(agent_config);
                    for i = 1:length(fields)
                        field = fields{i};
                        if isprop(agent, field)
                            agent.(field) = agent_config.(field);
                        end
                    end
                end
            end
        end
    end
end
