classdef ConfigManager < handle
    %% ConfigManager - FSP-TCS 配置管理器
    % ================================================================
    % 版本：v2.0 - 完整功能版本
    % 功能：
    % 1. 配置文件加载和保存
    % 2. 默认配置管理
    % 3. 配置验证和合并
    % 4. 嵌套配置访问
    % 5. 配置显示和导出
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
                    fprintf('⚠️  配置文件不存在，使用默认配置: %s\n', config_file);
                    config = ConfigManager.getDefaultConfig();
                end
                
            catch ME
                fprintf('❌ 配置文件加载失败: %s\n', ME.message);
                fprintf('💡 使用默认配置\n');
                config = ConfigManager.getDefaultConfig();
            end
        end
        
        function config = getDefaultConfig()
            %% 获取默认配置
            % 输出: config - 默认配置结构体
            
            config = struct();
            
            % =========================
            % 实验配置
            % =========================
            config.experiment = struct();
            config.experiment.name = 'FSP-TCS-IDS';
            config.experiment.version = '2.0';
            config.experiment.description = 'FSP-based Train Control System Intrusion Detection';
            config.experiment.timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
            
            % =========================
            % 系统配置
            % =========================
            config.system = struct();
            config.system.n_stations = 10;              % 主站数量
            config.system.components_per_station = 3;    % 每个主站的组件数
            config.system.n_components_per_station = repmat(3, 1, 10);  % 每个主站的组件数数组
            config.system.total_components = 30;         % 总组件数
            config.system.state_space_size = 77;         % 状态空间大小
            config.system.action_space_size = 15;        % 动作空间大小
            config.system.total_resources = 100;         % 总资源预算
            config.system.n_resource_types = 5;          % 资源类型数量
            config.system.n_attack_types = 6;            % 攻击类型数量
            
            % =========================
            % 仿真配置
            % =========================
            config.simulation = struct();
            config.simulation.n_iterations = 50;         % FSP迭代次数
            config.simulation.n_episodes_per_iter = 100; % 每次迭代的episodes数
            config.simulation.max_episode_steps = 100;   % 每个episode的最大步数
            config.simulation.alpha_ewma = 0.1;           % EWMA平滑参数
            config.simulation.max_time_hours = 24;       % 最大运行时间（小时）
            config.simulation.save_interval = 10;        % 保存间隔（迭代数）
            config.simulation.checkpoint_enabled = true; % 启用检查点
            
            % =========================
            % 学习配置
            % =========================
            config.learning = struct();
            config.learning.learning_rate = 0.1;         % 学习率
            config.learning.discount_factor = 0.9;       % 折扣因子
            config.learning.epsilon = 0.1;               % ε-贪心探索率
            config.learning.epsilon_min = 0.01;          % 最小探索率
            config.learning.epsilon_decay = 0.995;       % 探索率衰减
            config.learning.memory_size = 10000;         % 经验回放大小
            config.learning.batch_size = 32;             % 批处理大小
            config.learning.update_target_freq = 100;    % 目标网络更新频率
            
            % =========================
            % 算法配置
            % =========================
            config.algorithms = struct();
            config.algorithms.attacker = 'QLearning';    % 攻击者算法
            config.algorithms.defender = {'QLearning', 'SARSA', 'DoubleQLearning'}; % 防御者算法
            config.algorithms.use_dqn = false;           % 是否使用DQN（如不可用则用QLearning）
            config.algorithms.parallel_agents = true;    % 并行运行多个智能体
            
            % =========================
            % 环境配置
            % =========================
            config.environment = struct();
            config.environment.attack_success_rate = 0.3; % 攻击成功率
            config.environment.detection_accuracy = 0.85; % 检测准确率
            config.environment.false_positive_rate = 0.05; % 误报率
            config.environment.system_recovery_time = 5;   % 系统恢复时间
            config.environment.noise_level = 0.1;          % 环境噪声水平
            
            % =========================
            % 输出配置
            % =========================
            config.output = struct();
            config.output.save_results = true;           % 保存结果
            config.output.generate_reports = true;       % 生成报告
            config.output.log_file = 'simulation.log';   % 日志文件
            config.output.results_dir = 'results';       % 结果目录
            config.output.reports_dir = 'reports';       % 报告目录
            config.output.models_dir = 'models';         % 模型目录
            config.output.plots_format = 'png';          % 图表格式
            config.output.save_checkpoints = true;       % 保存检查点
            
            % =========================
            % 性能配置
            % =========================
            config.performance = struct();
            config.performance.use_parallel = true;      % 使用并行计算
            config.performance.max_workers = 4;          % 最大并行工作数
            config.performance.memory_limit_gb = 8;      % 内存限制（GB）
            config.performance.progress_update_freq = 10; % 进度更新频率
            
            % =========================
            % 调试配置
            % =========================
            config.debug = struct();
            config.debug.debug_mode = false;             % 调试模式
            config.debug.verbose = true;                 % 详细输出
            config.debug.plot_real_time = false;         % 实时绘图
            config.debug.save_debug_data = false;        % 保存调试数据
            config.debug.log_level = 'INFO';             % 日志级别 (DEBUG, INFO, WARNING, ERROR)
            
            % =========================
            % 兼容性字段（为了向后兼容）
            % =========================
            config.n_stations = config.system.n_stations;
            config.n_episodes_per_iter = config.simulation.n_episodes_per_iter;
            config.state_space_size = config.system.state_space_size;
            config.steps_per_episode = config.simulation.max_episode_steps;  % 使用max_episode_steps
            config.debug_mode = config.debug.debug_mode;
            config.n_components_per_station = config.system.n_components_per_station;  % 添加此字段
            config.total_resources = config.system.total_resources;  % 添加此字段
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
                if ~isfield(config, 'system')
                    config.system = struct();
                end
                if ~isfield(config, 'simulation')
                    config.simulation = struct();
                end
                if ~isfield(config, 'learning')
                    config.learning = struct();
                end
                if ~isfield(config, 'algorithms')
                    config.algorithms = struct();
                end
                
                % 验证基本数值参数
                if ~isfield(config.system, 'n_stations') || config.system.n_stations <= 0
                    warning('n_stations必须为正整数，设置为默认值10');
                    config.system.n_stations = 10;
                end
                
                if ~isfield(config.simulation, 'n_iterations') || config.simulation.n_iterations <= 0
                    warning('n_iterations必须为正整数，设置为默认值50');
                    config.simulation.n_iterations = 50;
                end
                
                if ~isfield(config.learning, 'learning_rate') || config.learning.learning_rate <= 0 || config.learning.learning_rate > 1
                    warning('learning_rate必须在(0,1]范围内，设置为默认值0.1');
                    config.learning.learning_rate = 0.1;
                end
                
                if ~isfield(config.learning, 'discount_factor') || config.learning.discount_factor < 0 || config.learning.discount_factor > 1
                    warning('discount_factor必须在[0,1]范围内，设置为默认值0.9');
                    config.learning.discount_factor = 0.9;
                end
                
                % 验证算法列表
                valid_algorithms = {'QLearning', 'SARSA', 'DoubleQLearning', 'DQN'};
                
                % 检查攻击者算法
                if ~isfield(config.algorithms, 'attacker')
                    config.algorithms.attacker = 'QLearning';
                else
                    % 处理可能的算法名称变体
                    attacker_alg = config.algorithms.attacker;
                    if ischar(attacker_alg)
                        % 标准化算法名称
                        switch lower(attacker_alg)
                            case {'qlearning', 'q-learning', 'q_learning'}
                                config.algorithms.attacker = 'QLearning';
                            case {'sarsa'}
                                config.algorithms.attacker = 'SARSA';
                            case {'doubleqlearning', 'double_q_learning', 'double-q-learning'}
                                config.algorithms.attacker = 'DoubleQLearning';
                            case {'dqn', 'deep_q_network'}
                                config.algorithms.attacker = 'DQN';
                            otherwise
                                if ~ismember(attacker_alg, valid_algorithms)
                                    warning('未知的攻击者算法: %s，设置为QLearning', attacker_alg);
                                    config.algorithms.attacker = 'QLearning';
                                end
                        end
                    else
                        warning('攻击者算法配置格式错误，设置为QLearning');
                        config.algorithms.attacker = 'QLearning';
                    end
                end
                
                % 验证防御者算法列表
                if ~isfield(config.algorithms, 'defender') || isempty(config.algorithms.defender)
                    config.algorithms.defender = {'QLearning'};
                elseif ~iscell(config.algorithms.defender)
                    config.algorithms.defender = {config.algorithms.defender};
                end
                
                % 验证每个防御者算法
                valid_defenders = {};
                for i = 1:length(config.algorithms.defender)
                    if ismember(config.algorithms.defender{i}, valid_algorithms)
                        valid_defenders{end+1} = config.algorithms.defender{i}; %#ok<AGROW>
                    else
                        warning('未知的防御者算法: %s，已移除', config.algorithms.defender{i});
                    end
                end
                
                % 如果防御者算法列表为空，使用默认
                if isempty(valid_defenders)
                    config.algorithms.defender = {'QLearning'};
                else
                    config.algorithms.defender = valid_defenders;
                end
                
                % 验证组件配置
                if ~isfield(config.system, 'n_components_per_station')
                    config.system.n_components_per_station = repmat(3, 1, config.system.n_stations);
                elseif ~isnumeric(config.system.n_components_per_station)
                    config.system.n_components_per_station = repmat(3, 1, config.system.n_stations);
                elseif length(config.system.n_components_per_station) ~= config.system.n_stations
                    % 如果长度不匹配，截断或扩展
                    if length(config.system.n_components_per_station) > config.system.n_stations
                        config.system.n_components_per_station = config.system.n_components_per_station(1:config.system.n_stations);
                    else
                        % 扩展数组
                        missing = config.system.n_stations - length(config.system.n_components_per_station);
                        config.system.n_components_per_station = [config.system.n_components_per_station, repmat(3, 1, missing)];
                    end
                end
                
                % 更新兼容性字段
                config.n_stations = config.system.n_stations;
                if isfield(config.simulation, 'n_episodes_per_iter')
                    config.n_episodes_per_iter = config.simulation.n_episodes_per_iter;
                else
                    config.n_episodes_per_iter = 100;
                end
                
                if isfield(config.system, 'state_space_size')
                    config.state_space_size = config.system.state_space_size;
                else
                    config.state_space_size = 77;
                end
                
                if isfield(config.simulation, 'max_episode_steps')
                    config.steps_per_episode = config.simulation.max_episode_steps;
                elseif isfield(config.simulation, 'steps_per_episode')
                    config.steps_per_episode = config.simulation.steps_per_episode;
                else
                    config.steps_per_episode = 100;
                    if ~isfield(config, 'simulation')
                        config.simulation = struct();
                    end
                    config.simulation.max_episode_steps = 100;
                end
                
                if isfield(config, 'debug') && isfield(config.debug, 'debug_mode')
                    config.debug_mode = config.debug.debug_mode;
                else
                    config.debug_mode = false;
                end
                
                config.n_components_per_station = config.system.n_components_per_station;
                
                if isfield(config.system, 'total_resources')
                    config.total_resources = config.system.total_resources;
                else
                    config.total_resources = 100;
                    config.system.total_resources = 100;
                end
                
            catch ME
                warning('配置验证过程中发生错误: %s', ME.message);
            end
        end
        
        function saveConfig(config, config_path)
            %% 保存配置到文件
            % 输入: config - 配置结构体
            %       config_path - 保存路径（可选）
            
            if nargin < 2
                config_path = ConfigManager.getDefaultConfigPath();
            end
            
            try
                % 确保目录存在
                config_dir = fileparts(config_path);
                if ~exist(config_dir, 'dir')
                    mkdir(config_dir);
                end
                
                % 转换为JSON并保存
                config_json = jsonencode(config, 'PrettyPrint', true);
                
                fid = fopen(config_path, 'w', 'n', 'UTF-8');
                if fid == -1
                    error('无法打开文件: %s', config_path);
                end
                
                fprintf(fid, '%s', config_json);
                fclose(fid);
                fprintf('✓ 配置已保存到: %s\n', config_path);
                
            catch ME
                error('配置保存失败: %s', ME.message);
            end
        end
        
        function displayConfig(config)
            %% 显示配置摘要
            % 输入: config - 配置结构体
            
            fprintf('\n=== FSP-TCS 仿真配置摘要 ===\n');
            fprintf('实验名称: %s (v%s)\n', config.experiment.name, config.experiment.version);
            fprintf('时间戳: %s\n', config.experiment.timestamp);
            
            fprintf('\n--- 系统配置 ---\n');
            fprintf('主站数量: %d\n', config.system.n_stations);
            fprintf('总组件数: %d\n', config.system.total_components);
            fprintf('状态空间大小: %d\n', config.system.state_space_size);
            fprintf('动作空间大小: %d\n', config.system.action_space_size);
            
            fprintf('\n--- 仿真配置 ---\n');
            fprintf('FSP迭代次数: %d\n', config.simulation.n_iterations);
            fprintf('每次迭代episodes: %d\n', config.simulation.n_episodes_per_iter);
            fprintf('每episode步数: %d\n', config.simulation.steps_per_episode);
            
            fprintf('\n--- 学习配置 ---\n');
            fprintf('学习率: %.3f\n', config.learning.learning_rate);
            fprintf('折扣因子: %.3f\n', config.learning.discount_factor);
            fprintf('探索率: %.3f (衰减: %.3f)\n', config.learning.epsilon, config.learning.epsilon_decay);
            
            fprintf('\n--- 算法配置 ---\n');
            fprintf('攻击者算法: %s\n', config.algorithms.attacker);
            fprintf('防御者算法: %s\n', strjoin(config.algorithms.defender, ', '));
            
            fprintf('\n--- 输出配置 ---\n');
            fprintf('保存结果: %s\n', mat2str(config.output.save_results));
            fprintf('生成报告: %s\n', mat2str(config.output.generate_reports));
            fprintf('结果目录: %s\n', config.output.results_dir);
            
            fprintf('===========================\n\n');
        end
        
        function value = getConfigValue(config, field_path, default_value)
            %% 获取嵌套配置值的辅助函数
            % 输入: config - 配置结构体
            %       field_path - 字段路径（如 'learning.learning_rate'）
            %       default_value - 默认值
            % 输出: value - 配置值
            
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
        
        function config_path = getDefaultConfigPath()
            %% 获取默认配置文件路径
            % 输出: config_path - 配置文件完整路径
            
            % 获取当前文件所在目录
            current_dir = fileparts(mfilename('fullpath'));
            config_path = fullfile(current_dir, ConfigManager.DEFAULT_CONFIG_FILE);
        end
        
        function merged = deepMergeStruct(struct1, struct2)
            %% 深度合并两个结构体
            % 输入: struct1 - 基础结构体（默认配置）
            %       struct2 - 覆盖结构体（用户配置）
            % 输出: merged - 合并后的结构体
            
            merged = struct1;
            
            if ~isstruct(struct2)
                return;
            end
            
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
        
        function exportConfigTemplate(output_path)
            %% 导出配置模板文件
            % 输入: output_path - 输出路径（可选）
            
            if nargin < 1
                output_path = 'config_template.json';
            end
            
            % 获取默认配置
            template_config = ConfigManager.getDefaultConfig();
            
            % 添加注释说明
            template_config.README = struct();
            template_config.README.description = 'FSP-TCS Configuration Template';
            template_config.README.usage = 'Modify values as needed and save as your_config.json';
            template_config.README.load_command = 'config = ConfigManager.loadConfig(''your_config.json'')';
            
            % 保存模板
            ConfigManager.saveConfig(template_config, output_path);
            fprintf('✓ 配置模板已导出到: %s\n', output_path);
        end
    end
end