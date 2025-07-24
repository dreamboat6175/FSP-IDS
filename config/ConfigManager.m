classdef ConfigManager < handle
    %% ConfigManager - FSP-TCS 配置管理器 (v3.2 - 慢收敛优化版)
    % ================================================================
    % 版本：v3.2 - 默认参数已根据“慢收敛”和“充分探索”原则进行优化
    % 功能：
    % 1. 配置文件加载和保存
    % 2. 优化的默认配置管理，旨在避免策略早熟
    % 3. 配置验证和合并
    % 4. 嵌套配置访问
    % 5. 配置显示和导出
    % 6. 智能体配置合并方法
    % ================================================================
    
    properties (Access = private, Constant)
        % 默认配置文件路径
        DEFAULT_CONFIG_FILE = 'default_config.json';
        CONFIG_DIR = 'config';
    end
    
    methods (Static)
        function config = loadConfig(config_file)
            %% 加载配置文件
            % 输入: config_file - 配置文件路径（可选）
            % 输出: config - 配置结构体
            
            if nargin < 1 || isempty(config_file)
                config_file = ConfigManager.getDefaultConfigPath();
            end
            
            try
                if exist(config_file, 'file')
                    fprintf('📁 加载配置文件: %s\n', config_file);
                    config_text = fileread(config_file);
                    config = jsondecode(config_text);
                    
                    % 验证和补充配置
                    config = ConfigManager.validateAndMergeConfig(config);
                    fprintf('✓ 配置加载完成\n');
                else
                    fprintf('⚠️  配置文件不存在，使用优化后的默认配置: %s\n', config_file);
                    config = ConfigManager.getDefaultConfig();
                end
                
            catch ME
                fprintf('❌ 配置文件加载失败: %s\n', ME.message);
                fprintf('💡 使用优化后的默认配置\n');
                config = ConfigManager.getDefaultConfig();
            end
        end
        
        function config = getDefaultConfig()
            %% 获取默认配置 - 优化版本，旨在解决早熟收敛问题
            % 输出: config - 默认配置结构体
            
            config = struct();
            
            % =========================
            % 实验配置
            % =========================
            config.experiment = struct();
            config.experiment.name = 'FSP-TCS-IDS';
            config.experiment.version = '3.2-slow-convergence';
            config.experiment.description = 'FSP-based Train Control System with RL optimized for exploration';
            config.experiment.timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
            
            % =========================
            % 系统配置
            % =========================
            config.system = struct();
            config.system.n_stations = 10;
            config.system.n_components_per_station = [7, 6, 8, 5, 9, 15, 4, 6, 3, 4];
            config.system.total_components = sum(config.system.n_components_per_station);
            config.system.total_resources = 100;
            config.system.n_resource_types = 5;
            config.system.n_attack_types = 6;
            config.system.state_space_size = 77;
            config.system.action_space_size = 101;
            
            % =========================
            % 仿真配置
            % =========================
            config.simulation = struct();
            config.simulation.n_iterations = 200; 
            config.simulation.n_episodes_per_iter = 100;
            config.simulation.max_episode_steps = 50;
            config.simulation.pool_size_limit = 50;
            config.simulation.pool_update_interval = 10;
            config.simulation.alpha_ewma = 0.1;
            
            % =========================
            % 学习配置 - 针对“慢收敛”和“充分探索”进行优化
            % =========================
            config.learning = struct();
            
            % 基础学习参数
            config.learning.learning_rate = 0.1;            % 适度降低初始学习率，使更新更平稳
            config.learning.discount_factor = 0.95;
            config.learning.exploration_strategy = 'epsilon-greedy';
            
            % Epsilon-Greedy 参数 - 核心优化，避免早熟收敛
            config.learning.epsilon = 0.8;                   % 提高初始探索率
            config.learning.epsilon_min = 0.2;               % 提高最小探索率
            config.learning.epsilon_decay = 0.99998;          % 减缓探索衰减

            % Softmax/Boltzmann 参数
            config.learning.temperature = 3.0;               % 提高初始温度
            config.learning.temperature_min = 1.0;           % 提高最小温度
            config.learning.temperature_decay = 0.9999;      % 减缓温度衰减

            % 学习率调度参数 - 优化
            config.learning.learning_rate_min = 0.1;        % 提高最小学习率
            config.learning.learning_rate_decay = 0.99995;   % 大幅减缓学习率衰减
            
            % 策略多样性参数
            config.learning.strategy_diversity_bonus = 0.05; % 策略多样性奖励
            config.learning.exploration_bonus = 0.02;        % 探索奖励
            config.learning.adaptive_epsilon = true;         % 自适应epsilon
            config.learning.min_exploration_episodes = 200;  % 最小探索回合数
            
            % =========================
            % 强化学习默认参数组合 - 新增结构
            % =========================
            config.rl_defaults = struct();
            config.rl_defaults.learning_rate = config.learning.learning_rate;
            config.rl_defaults.discount_factor = config.learning.discount_factor;
            config.rl_defaults.exploration_strategy = config.learning.exploration_strategy;
            
            % Epsilon-Greedy 优化参数
            config.rl_defaults.epsilon_greedy = struct();
            config.rl_defaults.epsilon_greedy.epsilon = config.learning.epsilon;
            config.rl_defaults.epsilon_greedy.epsilon_decay = config.learning.epsilon_decay;
            config.rl_defaults.epsilon_greedy.epsilon_min = config.learning.epsilon_min;
            
            % Softmax 优化参数
            config.rl_defaults.softmax_exploration = struct();
            config.rl_defaults.softmax_exploration.temperature = config.learning.temperature;
            config.rl_defaults.softmax_exploration.temperature_decay = config.learning.temperature_decay;
            config.rl_defaults.softmax_exploration.temperature_min = config.learning.temperature_min;
            
            % =========================
            % 算法配置
            % =========================
            config.algorithms = struct();
            config.algorithms.attacker = 'QLearning';
            config.algorithms.defender = {'QLearning', 'SARSA', 'DoubleQLearning'};
            
            % 算法特定参数
            config.algorithms.qlearning = struct();
            config.algorithms.qlearning.exploration_strategy = 'epsilon-greedy';
            config.algorithms.qlearning.use_optimized_params = true;
            
            config.algorithms.sarsa = struct();
            config.algorithms.sarsa.exploration_strategy = 'epsilon-greedy';
            config.algorithms.sarsa.use_optimized_params = true;
            
            config.algorithms.doubleqlearning = struct();
            config.algorithms.doubleqlearning.exploration_strategy = 'epsilon-greedy';
            config.algorithms.doubleqlearning.use_optimized_params = true;
            
            % =========================
            % 攻击配置
            % =========================
            config.attacks = struct();
            config.attacks.frequency = 0.3;
            config.attacks.success_probability = 0.4;
            config.attacks.types = {'malware', 'dos', 'intrusion', 'spoofing', 'tampering', 'eavesdropping'};
            config.attacks.severity = [0.8, 0.7, 0.6, 0.5, 0.9, 0.4];
            config.attacks.detection_difficulty = [0.6, 0.5, 0.7, 0.8, 0.4, 0.9];
            
            % =========================
            % 资源配置
            % =========================
            config.resources = struct();
            config.resources.types = {'computational', 'network', 'storage', 'security', 'backup'};
            config.resources.effectiveness = [0.7, 0.8, 0.6, 0.9, 0.5];
            config.resources.cost = [1.0, 1.2, 0.8, 1.5, 0.6];
            
            % =========================
            % 输出配置
            % =========================
            config.output = struct();
            config.output.save_results = true;
            config.output.save_figures = true;
            config.output.log_file = 'simulation.log';
            config.output.results_dir = 'results';
            config.output.figures_dir = 'figures';
            config.output.verbose = true;
            
            % =========================
            % 调试配置
            % =========================
            config.debug = struct();
            config.debug.debug_mode = false;
            config.debug.save_debug_data = false;
            config.debug.plot_convergence = true;
            config.debug.log_level = 'INFO';
            
            % =========================
            % 向后兼容性字段
            % =========================
            config = ConfigManager.ensureBackwardCompatibility(config);
        end
        
        function config = validateAndMergeConfig(user_config)
            %% 验证并合并用户配置与默认配置
            % 输入: user_config - 用户配置
            % 输出: config - 合并后的配置
            
            % 获取默认配置
            default_config = ConfigManager.getDefaultConfig();
            
            % 深度合并配置
            config = ConfigManager.deepMergeStruct(default_config, user_config);
            
            % 验证关键参数
            config = ConfigManager.validateConfig(config);
            
            fprintf('✓ 配置验证和合并完成\n');
        end
        
        function config = validateConfig(config)
            %% 验证配置参数的有效性
            % 输入/输出: config - 配置结构体
            
            try
                % 确保基本结构存在
                required_fields = {'system', 'simulation', 'learning', 'algorithms', 'output', 'debug'};
                for i = 1:length(required_fields)
                    field = required_fields{i};
                    if ~isfield(config, field) || ~isstruct(config.(field))
                        config.(field) = struct();
                    end
                end
                
                % 验证系统参数
                if config.system.n_stations <= 0
                    warning('n_stations 必须大于0，重置为10');
                    config.system.n_stations = 10;
                end
                
                % 验证学习参数
                if config.learning.epsilon < 0 || config.learning.epsilon > 1
                    warning('epsilon 必须在[0,1]范围内，重置为1.0');
                    config.learning.epsilon = 1.0;
                end
                
                if config.learning.epsilon_decay > 0.9999 || config.learning.epsilon_decay < 0.9
                     warning('epsilon_decay 推荐在[0.9, 0.9999]之间，以确保合理的衰减速度');
                end

                % 验证算法配置
                valid_algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
                if ~ismember(config.algorithms.attacker, valid_algorithms)
                    warning('未知攻击者算法，重置为QLearning');
                    config.algorithms.attacker = 'QLearning';
                end
                
                % 确保向后兼容性
                config = ConfigManager.ensureBackwardCompatibility(config);
                
                fprintf('✓ 配置验证完成\n');
                
            catch ME
                warning('配置验证过程中出错: %s', ME.message);
            end
        end
        
        function config = ensureBackwardCompatibility(config)
            %% 确保向后兼容性
            % 将新格式的配置映射到旧格式字段
            
            % 系统参数
            config.n_stations = config.system.n_stations;
            config.total_components = config.system.total_components;
            config.n_components_per_station = config.system.n_components_per_station;
            config.state_space_size = config.system.state_space_size;
            config.total_resources = config.system.total_resources;

            % 仿真参数
            config.n_iterations = config.simulation.n_iterations;
            config.n_episodes_per_iter = config.simulation.n_episodes_per_iter;
            config.steps_per_episode = config.simulation.max_episode_steps;
            config.pool_size_limit = config.simulation.pool_size_limit;
            config.pool_update_interval = config.simulation.pool_update_interval;
            config.alpha_ewma = config.simulation.alpha_ewma;

            % 学习参数
            config.learning_rate = config.learning.learning_rate;
            config.discount_factor = config.learning.discount_factor;
            config.epsilon = config.learning.epsilon;
            config.epsilon_min = config.learning.epsilon_min;
            config.epsilon_decay = config.learning.epsilon_decay;
            
            % 攻击参数
            config.attack_frequency = config.attacks.frequency;
            config.attack_success_probability = config.attacks.success_probability;
            config.attack_types = config.attacks.types;
            config.attack_severity = config.attacks.severity;
            config.attack_detection_difficulty = config.attacks.detection_difficulty;
            
            % 调试参数
            config.debug_mode = config.debug.debug_mode;
        end
        
        function merged = deepMergeStruct(struct1, struct2)
            %% 深度合并两个结构体
            % 输入: struct1 - 基础结构体（默认配置）
            %       struct2 - 覆盖结构体（用户配置）
            % 输出: merged - 合并后的结构体
            
            merged = struct1;
            fields = fieldnames(struct2);
            
            for i = 1:length(fields)
                field = fields{i};
                if isfield(merged, field) && isstruct(merged.(field)) && isstruct(struct2.(field))
                    % 递归合并子结构体
                    merged.(field) = ConfigManager.deepMergeStruct(merged.(field), struct2.(field));
                else
                    % 直接覆盖
                    merged.(field) = struct2.(field);
                end
            end
        end
        
        function path = getDefaultConfigPath()
            %% 获取默认配置文件路径
            path = fullfile(ConfigManager.CONFIG_DIR, ConfigManager.DEFAULT_CONFIG_FILE);
        end
        
        function saveConfig(config, filename)
            %% 保存配置到文件
            % 输入: config - 配置结构体
            %       filename - 文件名（可选）
            
            if nargin < 2
                filename = ConfigManager.DEFAULT_CONFIG_FILE;
            end
            
            config_path = fullfile(ConfigManager.CONFIG_DIR, filename);
            
            % 确保目录存在
            config_dir = fileparts(config_path);
            if ~exist(config_dir, 'dir')
                mkdir(config_dir);
            end
            
            try
                config_json = jsonencode(config, 'PrettyPrint', true);
                fid = fopen(config_path, 'w', 'n', 'UTF-8');
                fprintf(fid, '%s', config_json);
                fclose(fid);
                fprintf('✓ 配置已保存到: %s\n', config_path);
            catch ME
                error('配置保存失败: %s', ME.message);
            end
        end
        
        function displayConfig(config)
            %% 显示配置摘要
            
            fprintf('\n=== FSP-TCS 仿真配置摘要 (v3.2 - 慢收敛优化版) ===\n');
            fprintf('实验名称: %s v%s\n', config.experiment.name, config.experiment.version);
            fprintf('系统配置: %d个主站, %d个组件\n', ...
                   config.system.n_stations, config.system.total_components);
            fprintf('仿真参数: %d次迭代, 每次%d个episodes\n', ...
                   config.simulation.n_iterations, config.simulation.n_episodes_per_iter);
            
            fprintf('\n--- 优化的学习参数 (旨在充分探索) ---\n');
            fprintf('学习率: %.3f (最小: %.3f, 衰减: %.4f)\n', ...
                   config.learning.learning_rate, config.learning.learning_rate_min, config.learning.learning_rate_decay);
            fprintf('探索策略: %s\n', config.learning.exploration_strategy);
            fprintf('Epsilon: %.3f → %.3f (衰减: %.4f) [慢衰减]\n', ...
                   config.learning.epsilon, config.learning.epsilon_min, config.learning.epsilon_decay);
            fprintf('温度参数: %.3f → %.3f (衰减: %.4f)\n', ...
                   config.learning.temperature, config.learning.temperature_min, config.learning.temperature_decay);
            
            fprintf('\n--- 算法配置 ---\n');
            fprintf('攻击者算法: %s\n', config.algorithms.attacker);
            fprintf('防御者算法: %s\n', strjoin(config.algorithms.defender, ', '));
            
            fprintf('\n--- 输出配置 ---\n');
            fprintf('结果保存: %s, 图形保存: %s\n', ...
                   mat2str(config.output.save_results), mat2str(config.output.save_figures));
            fprintf('日志文件: %s\n', config.output.log_file);
            fprintf('======================================================\n\n');
        end
        
        function value = getConfigValue(config, field_path, default_value)
            %% 获取嵌套配置值的辅助函数
            % 输入: config - 配置结构体
            %       field_path - 字段路径（如 'learning.learning_rate'）
            %       default_value - 默认值
            
            if nargin < 3
                default_value = [];
            end
            
            try
                path_parts = strsplit(field_path, '.');
                value = config;
                for i = 1:length(path_parts)
                    if isfield(value, path_parts{i})
                        value = value.(path_parts{i});
                    else
                        value = default_value;
                        return;
                    end
                end
            catch
                value = default_value;
            end
        end
        
        function mergeAgentConfig(agent, config)
            %% 合并智能体配置 - 使用优化后的默认值
            % 输入: agent - 智能体对象
            %       config - 配置结构体
            
            try
                fprintf('🔧 正在为智能体配置优化的学习参数: %s\n', agent.name);
                
                % 使用 config.learning 中的优化参数
                agent.learning_rate = ConfigManager.safeGetConfigValue(config.learning, 'learning_rate', 0.1);
                agent.discount_factor = ConfigManager.safeGetConfigValue(config.learning, 'discount_factor', 0.95);
                agent.exploration_strategy = ConfigManager.safeGetConfigValue(config.learning, 'exploration_strategy', 'epsilon-greedy');
                
                agent.epsilon = ConfigManager.safeGetConfigValue(config.learning, 'epsilon', 1.0);
                agent.epsilon_min = ConfigManager.safeGetConfigValue(config.learning, 'epsilon_min', 0.15);
                agent.epsilon_decay = ConfigManager.safeGetConfigValue(config.learning, 'epsilon_decay', 0.9999);
                
                agent.temperature = ConfigManager.safeGetConfigValue(config.learning, 'temperature', 5.0);
                agent.temperature_decay = ConfigManager.safeGetConfigValue(config.learning, 'temperature_decay', 0.9999);
                agent.temperature_min = ConfigManager.safeGetConfigValue(config.learning, 'temperature_min', 0.5);
                
                agent.learning_rate_min = ConfigManager.safeGetConfigValue(config.learning, 'learning_rate_min', 0.01);
                agent.learning_rate_decay = ConfigManager.safeGetConfigValue(config.learning, 'learning_rate_decay', 0.9999);
                
                agent.pool_size_limit = ConfigManager.safeGetConfigValue(config.simulation, 'pool_size_limit', 50);
                
                if isprop(agent, 'strategy_diversity_bonus')
                    agent.strategy_diversity_bonus = ConfigManager.safeGetConfigValue(config.learning, 'strategy_diversity_bonus', 0.05);
                end
                if isprop(agent, 'exploration_bonus')
                    agent.exploration_bonus = ConfigManager.safeGetConfigValue(config.learning, 'exploration_bonus', 0.02);
                end
                
                fprintf('✓ 智能体配置完成: %s (ε=%.3f→%.3f, lr=%.3f→%.3f)\n', ...
                        agent.name, agent.epsilon, agent.epsilon_min, ...
                        agent.learning_rate, agent.learning_rate_min);
                
            catch ME
                fprintf('⚠️ 智能体配置合并失败: %s，使用硬编码的优化默认配置\n', ME.message);
                ConfigManager.setOptimizedAgentDefaults(agent);
            end
        end
        
        function setOptimizedAgentDefaults(agent)
            %% 为智能体设置硬编码的优化默认配置值（作为后备）
            
            agent.learning_rate = 0.1;
            agent.discount_factor = 0.95;
            agent.exploration_strategy = 'epsilon-greedy';
            
            agent.epsilon = 1.0;
            agent.epsilon_min = 0.15;
            agent.epsilon_decay = 0.9999;
            
            agent.temperature = 5.0;
            agent.temperature_decay = 0.9999;
            agent.temperature_min = 0.5;
            
            agent.learning_rate_min = 0.01;
            agent.learning_rate_decay = 0.9999;
            
            agent.pool_size_limit = 50;
            
            fprintf('✓ 已应用后备的优化默认配置到智能体: %s\n', agent.name);
        end
        
        function exportOptimizedTemplate(output_path)
            %% 导出优化的配置模板文件
            % 输入: output_path - 输出路径（可选）
            
            if nargin < 1
                output_path = 'optimized_config_template.json';
            end
            
            template_config = ConfigManager.getDefaultConfig();
            
            template_config.README = struct();
            template_config.README.description = 'FSP-TCS 优化配置模板 - 旨在解决早熟收敛问题';
            template_config.README.key_optimizations = {
                'epsilon初始值为1.0，进行完全探索',
                '极大地减缓epsilon和学习率的衰减速度 (0.9999)',
                '保持较高的epsilon_min (0.15)以避免后期探索停滞',
                '增加仿真迭代次数以适应更长的学习过程'
            };
            template_config.README.usage = '可基于此模板创建您自己的配置文件，或直接运行以使用优化后的默认值';
            template_config.README.load_command = 'config = ConfigManager.loadConfig(''your_config.json'')';
            
            ConfigManager.saveConfig(template_config, output_path);
            fprintf('✓ 优化配置模板已导出到: %s\n', output_path);
        end
    end
    
    methods (Access = private, Static)
        function value = safeGetConfigValue(conf, field, default_val)
            %% 安全的配置值获取辅助函数
            if isfield(conf, field) && ~isempty(conf.(field))
                value = conf.(field);
            else
                value = default_val;
            end
        end
    end
end