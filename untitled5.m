%% 应用Epsilon参数修复脚本 - apply_epsilon_fix.m
% =========================================================================
% 用途: 解决epsilon_min和epsilon_decay过小导致的策略固化问题
% 使用方法: 
%   1. 运行此脚本生成优化配置文件
%   2. 或者直接调用修复函数
% =========================================================================

function apply_epsilon_fix()
    %% 主修复函数
    fprintf('🔧 开始应用Epsilon参数修复...\n\n');
    
    % 1. 创建优化配置文件
    create_optimized_config();
    
    % 2. 修复现有ConfigManager（如果需要）
    fix_config_manager();
    
    % 3. 提供使用指导
    provide_usage_guidance();
    
    fprintf('✅ Epsilon参数修复完成！\n');
end

function create_optimized_config()
    %% 创建优化的配置文件
    fprintf('📝 正在创建优化配置文件...\n');
    
    % 获取优化配置结构
    config = get_optimized_config();
    
    % 保存为JSON文件
    config_path = fullfile('config', 'optimized_config.json');
    
    % 确保目录存在
    config_dir = fileparts(config_path);
    if ~exist(config_dir, 'dir')
        mkdir(config_dir);
    end
    
    try
        % 保存配置文件
        config_json = jsonencode(config, 'PrettyPrint', true);
        fid = fopen(config_path, 'w', 'n', 'UTF-8');
        fprintf(fid, '%s', config_json);
        fclose(fid);
        fprintf('✓ 优化配置已保存到: %s\n', config_path);
    catch ME
        fprintf('❌ 配置文件保存失败: %s\n', ME.message);
    end
end

function config = get_optimized_config()
    %% 获取优化的配置结构
    
    config = struct();
    
    % 元数据
    config.metadata = struct();
    config.metadata.version = '1.2';
    config.metadata.description = 'FSP-TCS 优化配置 - 解决策略固化问题';
    config.metadata.created = datestr(now, 'yyyy-mm-dd');
    config.metadata.optimizations = {
        '提高epsilon_min至0.15，防止完全停止探索';
        '将epsilon_decay调整至0.9995，减缓收敛速度';
        '增加temperature参数，增强Softmax探索';
        '优化学习率衰减，保持长期学习能力'
    };
    
    % 系统配置
    config.system = struct();
    config.system.n_stations = 10;
    config.system.n_components_per_station = [7, 6, 8, 5, 9, 15, 4, 6, 3, 4];
    config.system.total_components = 67;
    config.system.total_resources = 100;
    config.system.n_resource_types = 5;
    config.system.n_attack_types = 6;
    config.system.state_space_size = 77;
    config.system.action_space_size = 101;
    
    % 仿真配置
    config.simulation = struct();
    config.simulation.n_iterations = 1000;
    config.simulation.n_episodes_per_iter = 50;
    config.simulation.max_episode_steps = 50;
    config.simulation.pool_size_limit = 50;
    config.simulation.pool_update_interval = 10;
    config.simulation.alpha_ewma = 0.1;
    config.simulation.save_interval = 10;
    config.simulation.max_time_hours = 24;
    
    % 学习配置 - 关键优化部分
    config.learning = struct();
    config.learning.learning_rate = 0.15;          % 提高初始学习率
    config.learning.discount_factor = 0.95;
    config.learning.exploration_strategy = 'epsilon-greedy';
    
    % Epsilon参数优化 - 解决策略固化
    config.learning.epsilon = 0.5;                 % 提高初始探索率
    config.learning.epsilon_min = 0.15;            % 关键：提高最小探索率
    config.learning.epsilon_decay = 0.9995;        % 关键：减缓衰减速度
    
    % Softmax温度参数
    config.learning.temperature = 2.0;             % 提高初始温度
    config.learning.temperature_min = 0.5;         % 提高最小温度  
    config.learning.temperature_decay = 0.9997;    % 减缓温度衰减
    
    % 学习率调度优化
    config.learning.learning_rate_min = 0.05;      % 提高最小学习率
    config.learning.learning_rate_decay = 0.9998;  % 大幅减缓学习率衰减
    
    % 策略多样性机制
    config.learning.strategy_diversity_bonus = 0.05;
    config.learning.exploration_bonus = 0.02;
    config.learning.adaptive_epsilon = true;
    config.learning.min_exploration_episodes = 200;
    
    % 算法配置
    config.algorithms = struct();
    config.algorithms.attacker = 'QLearning';
    config.algorithms.defender = {'QLearning', 'SARSA', 'DoubleQLearning'};
    
    % 攻击配置
    config.attacks = struct();
    config.attacks.frequency = 0.3;
    config.attacks.success_probability = 0.7;
    config.attacks.types = {'DoS', 'malware', 'physical', 'social', 'insider', 'advanced'};
    config.attacks.severity = [0.2, 0.4, 0.6, 0.8, 1.0];
    config.attacks.detection_difficulty = [0.1, 0.3, 0.5, 0.7, 0.9];
    
    % 输出配置
    config.output = struct();
    config.output.save_results = true;
    config.output.save_plots = true;
    config.output.save_checkpoints = true;
    config.output.plot_interval = 50;
    config.output.verbose = true;
    
    % 调试配置
    config.debug = struct();
    config.debug.debug_mode = false;
    config.debug.log_level = 'INFO';
    config.debug.detailed_logging = false;
    
    % 强化学习默认参数
    config.rl_defaults = struct();
    config.rl_defaults.learning_rate = 0.15;
    config.rl_defaults.discount_factor = 0.95;
    config.rl_defaults.exploration_strategy = 'epsilon-greedy';
    
    config.rl_defaults.epsilon_greedy = struct();
    config.rl_defaults.epsilon_greedy.epsilon = 0.5;
    config.rl_defaults.epsilon_greedy.epsilon_decay = 0.9995;
    config.rl_defaults.epsilon_greedy.epsilon_min = 0.15;
    
    config.rl_defaults.softmax_exploration = struct();
    config.rl_defaults.softmax_exploration.temperature = 2.0;
    config.rl_defaults.softmax_exploration.temperature_decay = 0.9997;
    config.rl_defaults.softmax_exploration.temperature_min = 0.5;
end

function fix_config_manager()
    %% 如果需要，可以直接修复ConfigManager的默认值
    fprintf('🔧 检查ConfigManager是否需要修复...\n');
    
    % 检查ConfigManager文件是否存在
    config_manager_path = fullfile('config', 'ConfigManager.m');
    if ~exist(config_manager_path, 'file')
        fprintf('⚠️ ConfigManager.m文件未找到，跳过直接修复\n');
        return;
    end
    
    fprintf('✓ ConfigManager.m存在，建议使用优化配置文件\n');
end

function provide_usage_guidance()
    %% 提供使用指导
    fprintf('\n📋 使用指导:\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('1️⃣ 使用优化配置文件:\n');
    fprintf('   config = ConfigManager.loadConfig(''optimized_config.json'');\n\n');
    
    fprintf('2️⃣ 或者在main_fsp.m中修改配置加载:\n');
    fprintf('   %% 将第59行附近的配置加载改为:\n');
    fprintf('   config = ConfigManager.loadConfig(''optimized_config.json'');\n\n');
    
    fprintf('3️⃣ 关键参数说明:\n');
    fprintf('   • epsilon_min = 0.15   (防止探索完全停止)\n');
    fprintf('   • epsilon_decay = 0.9995 (减缓衰减，避免过快收敛)\n');
    fprintf('   • temperature = 2.0    (增强Softmax探索)\n');
    fprintf('   • learning_rate_min = 0.05 (保持学习能力)\n\n');
    
    fprintf('4️⃣ 预期效果:\n');
    fprintf('   ✓ 消除策略固化警告\n');
    fprintf('   ✓ 提高智能体探索多样性\n'); 
    fprintf('   ✓ 改善长期学习性能\n');
    fprintf('   ✓ 避免过早收敛到次优策略\n\n');
    
    fprintf('5️⃣ 验证方法:\n');
    fprintf('   运行仿真后检查日志，应该不再出现epsilon相关警告\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
end

function apply_agent_epsilon_fix(agent)
    %% 直接为智能体应用epsilon修复
    % 输入: agent - 智能体对象
    
    if nargin == 0
        fprintf('⚠️ 需要提供智能体对象作为输入参数\n');
        return;
    end
    
    try
        fprintf('🔧 正在为智能体 %s 应用epsilon修复...\n', agent.name);
        
        % 应用优化参数
        agent.epsilon = 0.5;           % 提高初始探索率
        agent.epsilon_min = 0.15;      % 关键修复：提高最小探索率
        agent.epsilon_decay = 0.9995;  % 关键修复：减缓衰减速度
        
        % 其他相关参数
        agent.learning_rate = 0.15;
        agent.learning_rate_min = 0.05;
        agent.learning_rate_decay = 0.9998;
        
        % 如果使用Softmax策略
        if isprop(agent, 'temperature') || isfield(agent, 'temperature')
            agent.temperature = 2.0;
            agent.temperature_min = 0.5;
            agent.temperature_decay = 0.9997;
        end
        
        fprintf('✓ 智能体 %s epsilon修复完成\n', agent.name);
        fprintf('  新参数: ε=%.3f→%.3f (衰减=%.4f)\n', ...
                agent.epsilon, agent.epsilon_min, agent.epsilon_decay);
        
    catch ME
        fprintf('❌ 智能体epsilon修复失败: %s\n', ME.message);
    end
end

function show_parameter_comparison()
    %% 显示参数对比
    fprintf('\n📊 参数对比 (旧值 → 新值):\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('epsilon_min:      0.01 → 0.15   (15倍提升，保持探索)\n');
    fprintf('epsilon_decay:    0.995 → 0.9995 (10倍减缓，避免过快收敛)\n');
    fprintf('learning_rate:    0.1 → 0.15     (提高学习效率)\n');
    fprintf('learning_rate_min: 0.001 → 0.05  (50倍提升，保持学习)\n');
    fprintf('temperature:      1.0 → 2.0      (双倍提升，增强探索)\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
end

%% 如果直接运行此脚本，执行主修复函数
if ~exist('calling_function', 'var')
    apply_epsilon_fix();
    show_parameter_comparison();
end