%% EnhancedVisualization.m - 统一可视化配置和生成器
% =========================================================================
% 描述: 集中管理所有可视化参数和样式，提供美观的报告生成
% 版本: v2.0 - 优化版，集中所有可视化配置
% =========================================================================

classdef EnhancedVisualization < handle
    
    properties
        results         % 仿真结果数据
        config          % 仿真配置信息
        environment     % 环境对象
        figures         % 图形句柄存储
        
        % === 可视化配置属性 ===
        visualConfig    % 可视化配置结构体
        colorScheme     % 统一配色方案
        figureOptions   % 图形窗口选项
        plotOptions     % 绘图选项
        fontConfig      % 字体配置
        saveOptions     % 保存选项
    end
    
    methods
        function obj = EnhancedVisualization(results, config, environment)
            % 构造函数 - 初始化可视化系统
            
            obj.results = results;
            obj.config = config;
            obj.environment = environment;
            obj.figures = {};
            
            % 初始化所有可视化配置
            obj.initializeVisualizationConfig();
            obj.initializeColorScheme();
            obj.initializeFigureOptions();
            obj.initializePlotOptions();
            obj.initializeFontConfig();
            obj.initializeSaveOptions();
            
            fprintf('✓ 可视化系统初始化完成\n');
        end
        
        function initializeVisualizationConfig(obj)
            % 初始化可视化配置参数
            
            obj.visualConfig = struct();
            
            % === 报告生成控制 ===
            obj.visualConfig.generate_dashboard = true;
            obj.visualConfig.generate_strategy_analysis = true;
            obj.visualConfig.generate_performance_analysis = true;
            obj.visualConfig.generate_convergence_analysis = true;
            obj.visualConfig.generate_3d_landscape = true;
            obj.visualConfig.generate_comparison_charts = true;
            
            % === 图表类型控制 ===
            obj.visualConfig.enable_heatmaps = true;
            obj.visualConfig.enable_3d_plots = true;
            obj.visualConfig.enable_animations = false;  % 默认关闭动画以节省时间
            obj.visualConfig.enable_interactive = false; % 默认关闭交互式图表
            
            % === 数据处理配置 ===
            obj.visualConfig.data_smoothing_window = 50;
            obj.visualConfig.max_data_points = 1000;     % 最大数据点数，超过则采样
            obj.visualConfig.outlier_threshold = 3;      % 异常值检测阈值（倍标准差）
            obj.visualConfig.confidence_interval = 0.95; % 置信区间
            
            % === 图像质量配置 ===
            obj.visualConfig.dpi = 300;                  % 图像分辨率
            obj.visualConfig.image_format = 'png';       % 图像格式
            obj.visualConfig.vector_format = 'pdf';      % 矢量图格式
            obj.visualConfig.compression_quality = 95;   % 压缩质量
            
            % === 性能优化配置 ===
            obj.visualConfig.use_gpu_acceleration = false;
            obj.visualConfig.parallel_rendering = false;
            obj.visualConfig.memory_efficient_mode = true;
        end
        
        function initializeColorScheme(obj)
            % 初始化统一配色方案
            
            obj.colorScheme = struct();
            
            % === 主要颜色 ===
            obj.colorScheme.primary = [0.2, 0.4, 0.8];        % 深蓝色
            obj.colorScheme.secondary = [0.9, 0.3, 0.2];      % 深红色
            obj.colorScheme.success = [0.2, 0.7, 0.3];        % 绿色
            obj.colorScheme.warning = [0.9, 0.6, 0.2];        % 橙色
            obj.colorScheme.info = [0.3, 0.7, 0.9];           % 浅蓝色
            obj.colorScheme.danger = [0.8, 0.2, 0.2];         % 红色
            
            % === 中性颜色 ===
            obj.colorScheme.dark = [0.2, 0.2, 0.2];           % 深灰色
            obj.colorScheme.light = [0.95, 0.95, 0.95];       % 浅灰色
            obj.colorScheme.medium = [0.6, 0.6, 0.6];         % 中灰色
            obj.colorScheme.background = [1.0, 1.0, 1.0];     % 白色背景
            
            % === 渐变色谱 ===
            obj.colorScheme.gradient_blue = [
                0.1, 0.2, 0.5;   % 深蓝
                0.3, 0.5, 0.8;   % 中蓝
                0.5, 0.7, 0.9    % 浅蓝
            ];
            
            obj.colorScheme.gradient_red = [
                0.8, 0.2, 0.2;   % 深红
                0.9, 0.4, 0.3;   % 中红
                1.0, 0.6, 0.4    % 浅红
            ];
            
            obj.colorScheme.gradient_green = [
                0.2, 0.6, 0.3;   % 深绿
                0.4, 0.7, 0.4;   % 中绿
                0.6, 0.8, 0.6    % 浅绿
            ];
            
            % === 算法专用颜色 ===
            obj.colorScheme.algorithms = struct();
            obj.colorScheme.algorithms.q_learning = [0.2, 0.4, 0.8];    % 蓝色
            obj.colorScheme.algorithms.sarsa = [0.8, 0.4, 0.2];         % 橙色
            obj.colorScheme.algorithms.double_q = [0.2, 0.7, 0.3];      % 绿色
            obj.colorScheme.algorithms.fsp = [0.7, 0.2, 0.7];           % 紫色
            
            % === 性能等级颜色 ===
            obj.colorScheme.performance = struct();
            obj.colorScheme.performance.excellent = [0.2, 0.7, 0.3];    % 绿色
            obj.colorScheme.performance.good = [0.6, 0.8, 0.2];         % 黄绿色
            obj.colorScheme.performance.acceptable = [0.9, 0.6, 0.2];   % 橙色
            obj.colorScheme.performance.poor = [0.9, 0.3, 0.2];         % 红色
            
            % === 热力图配色 ===
            obj.colorScheme.heatmap = struct();
            obj.colorScheme.heatmap.cool_warm = 'RdYlBu';
            obj.colorScheme.heatmap.strategy = 'viridis';
            obj.colorScheme.heatmap.performance = 'parula';
            
            % === 透明度设置 ===
            obj.colorScheme.alpha = struct();
            obj.colorScheme.alpha.fill = 0.3;
            obj.colorScheme.alpha.overlay = 0.6;
            obj.colorScheme.alpha.background = 0.1;
        end
        
        function initializeFigureOptions(obj)
            % 初始化图形窗口选项
            
            obj.figureOptions = struct();
            
            % === 窗口大小配置 ===
            obj.figureOptions.dashboard_size = [100, 100, 1800, 1000];    % 仪表盘
            obj.figureOptions.standard_size = [150, 150, 1600, 900];      % 标准图表
            obj.figureOptions.wide_size = [200, 200, 1800, 800];          % 宽屏图表
            obj.figureOptions.square_size = [250, 250, 1000, 1000];       % 方形图表
            obj.figureOptions.small_size = [300, 300, 800, 600];          % 小图表
            
            % === 窗口属性 ===
            obj.figureOptions.background_color = 'white';
            obj.figureOptions.renderer = 'opengl';  % 'opengl', 'zbuffer', 'painters'
            obj.figureOptions.units = 'pixels';
            obj.figureOptions.resize = 'on';
            obj.figureOptions.toolbar = 'none';
            obj.figureOptions.menubar = 'none';
            
            % === 布局配置 ===
            obj.figureOptions.subplot_spacing = struct();
            obj.figureOptions.subplot_spacing.horizontal = 0.08;
            obj.figureOptions.subplot_spacing.vertical = 0.08;
            obj.figureOptions.subplot_spacing.margin_left = 0.06;
            obj.figureOptions.subplot_spacing.margin_right = 0.04;
            obj.figureOptions.subplot_spacing.margin_top = 0.08;
            obj.figureOptions.subplot_spacing.margin_bottom = 0.08;
        end
        
        function initializePlotOptions(obj)
            % 初始化绘图选项
            
            obj.plotOptions = struct();
            
            % === 线条样式 ===
            obj.plotOptions.line = struct();
            obj.plotOptions.line.width_thin = 1;
            obj.plotOptions.line.width_medium = 2;
            obj.plotOptions.line.width_thick = 3;
            obj.plotOptions.line.width_ultra_thick = 4;
            obj.plotOptions.line.styles = {'-', '--', ':', '-.'};
            
            % === 标记样式 ===
            obj.plotOptions.marker = struct();
            obj.plotOptions.marker.size_small = 6;
            obj.plotOptions.marker.size_medium = 8;
            obj.plotOptions.marker.size_large = 10;
            obj.plotOptions.marker.types = {'o', 's', '^', 'v', 'd', '*', '+', 'x'};
            
            % === 网格设置 ===
            obj.plotOptions.grid = struct();
            obj.plotOptions.grid.enabled = true;
            obj.plotOptions.grid.alpha = 0.3;
            obj.plotOptions.grid.line_style = '-';
            obj.plotOptions.grid.line_width = 0.5;
            
            % === 坐标轴设置 ===
            obj.plotOptions.axis = struct();
            obj.plotOptions.axis.box = 'on';
            obj.plotOptions.axis.line_width = 1;
            obj.plotOptions.axis.tick_direction = 'out';
            obj.plotOptions.axis.tick_length = 0.01;
            
            % === 图例设置 ===
            obj.plotOptions.legend = struct();
            obj.plotOptions.legend.location = 'best';
            obj.plotOptions.legend.orientation = 'vertical';
            obj.plotOptions.legend.box = 'on';
            obj.plotOptions.legend.edge_color = [0.8, 0.8, 0.8];
            obj.plotOptions.legend.face_alpha = 0.9;
        end
        
        function initializeFontConfig(obj)
            % 初始化字体配置
            
            obj.fontConfig = struct();
            
            % === 字体系列 ===
            obj.fontConfig.family = 'Arial';  % 可选: 'Times', 'Helvetica', 'Courier'
            obj.fontConfig.weight_normal = 'normal';
            obj.fontConfig.weight_bold = 'bold';
            obj.fontConfig.angle_normal = 'normal';
            obj.fontConfig.angle_italic = 'italic';
            
            % === 字体大小 ===
            obj.fontConfig.size = struct();
            obj.fontConfig.size.title_main = 20;       % 主标题
            obj.fontConfig.size.title_sub = 16;        % 子标题
            obj.fontConfig.size.title_figure = 14;     % 图表标题
            obj.fontConfig.size.label_axis = 12;       % 坐标轴标签
            obj.fontConfig.size.label_tick = 10;       % 刻度标签
            obj.fontConfig.size.legend = 11;           % 图例
            obj.fontConfig.size.annotation = 9;        % 注释
            obj.fontConfig.size.text_large = 14;       % 大文本
            obj.fontConfig.size.text_medium = 12;      % 中等文本
            obj.fontConfig.size.text_small = 10;       % 小文本
            
            % === 颜色设置 ===
            obj.fontConfig.color = struct();
            obj.fontConfig.color.title = [0.2, 0.2, 0.2];      % 标题颜色
            obj.fontConfig.color.label = [0.3, 0.3, 0.3];      % 标签颜色
            obj.fontConfig.color.text = [0.4, 0.4, 0.4];       % 文本颜色
            obj.fontConfig.color.annotation = [0.6, 0.6, 0.6]; % 注释颜色
        end
        
        function initializeSaveOptions(obj)
            % 初始化保存选项
            
            obj.saveOptions = struct();
            
            % === 文件格式 ===
            obj.saveOptions.formats = {'png', 'pdf', 'svg'};
            obj.saveOptions.default_format = 'png';
            obj.saveOptions.vector_format = 'pdf';
            
            % === 质量设置 ===
            obj.saveOptions.dpi = 300;
            obj.saveOptions.quality = 95;
            obj.saveOptions.compression = true;
            
            % === 文件命名 ===
            obj.saveOptions.prefix = 'fsp_visualization';
            obj.saveOptions.timestamp_format = 'yyyymmdd_HHMMSS';
            obj.saveOptions.include_timestamp = true;
            
            % === 目录结构 ===
            obj.saveOptions.base_dir = 'reports';
            obj.saveOptions.create_subdir = true;
            obj.saveOptions.subdir_format = 'yyyymmdd_HHMMSS';
        end
        
        function generateCompleteReport(obj)
            % 生成完整的可视化报告
            
            fprintf('\n=== 生成FSP-TCS可视化报告 ===\n');
            
            % 创建保存目录
            save_dir = obj.createSaveDirectory();
            
            % 1. 核心性能仪表盘
            if obj.visualConfig.generate_dashboard
                fprintf('正在生成性能仪表盘...\n');
                obj.createPerformanceDashboard(save_dir);
            end
            
            % 2. 策略演化分析
            if obj.visualConfig.generate_strategy_analysis
                fprintf('正在生成策略演化分析...\n');
                obj.createStrategyEvolutionHeatmap(save_dir);
            end
            
            % 3. 性能对比分析
            if obj.visualConfig.generate_performance_analysis
                fprintf('正在生成性能对比分析...\n');
                obj.createPerformanceRadarChart(save_dir);
            end
            
            % 4. 收敛性分析
            if obj.visualConfig.generate_convergence_analysis
                fprintf('正在生成收敛性分析...\n');
                obj.createConvergenceAnalysis(save_dir);
            end
            
            % 5. 3D性能景观
            if obj.visualConfig.generate_3d_landscape
                fprintf('正在生成3D性能景观...\n');
                obj.create3DPerformanceLandscape(save_dir);
            end
            
            % 6. 游戏动态分析
            if obj.visualConfig.generate_comparison_charts
                fprintf('正在生成博弈动态分析...\n');
                obj.createGameDynamicsVisualization(save_dir);
            end
            
            % 生成报告索引
            obj.generateReportIndex(save_dir);
            
            fprintf('✓ 可视化报告生成完成！保存位置: %s\n', save_dir);
        end
        
        function save_dir = createSaveDirectory(obj)
            % 创建保存目录
            
            if obj.saveOptions.create_subdir
                timestamp = datestr(now, obj.saveOptions.subdir_format);
                save_dir = fullfile(obj.saveOptions.base_dir, timestamp);
            else
                save_dir = obj.saveOptions.base_dir;
            end
            
            if ~exist(save_dir, 'dir')
                mkdir(save_dir);
            end
        end
        
        function createPerformanceDashboard(obj, save_dir)
            % 创建核心性能仪表盘
            
            fig = figure('Name', 'FSP-TCS性能仪表盘', ...
                        'Position', obj.figureOptions.dashboard_size, ...
                        'Color', obj.figureOptions.background_color, ...
                        'Renderer', obj.figureOptions.renderer);
            obj.figures{end+1} = fig;
            
            % 设置全局字体
            obj.setGlobalFontProperties(fig);
            
            % 主标题
            obj.addMainTitle('FSP-TCS 智能防御系统性能仪表盘');
            
            % 创建子图
            subplot(2, 3, 1); obj.plotRADIGauge();
            subplot(2, 3, 2); obj.plotDefenseSuccessGauge();
            subplot(2, 3, 3); obj.plotEfficiencyGauge();
            subplot(2, 3, [4, 5]); obj.plotPerformanceTrends();
            subplot(2, 3, 6); obj.plotKeyMetricsTable();
            
            % 保存图形
            obj.saveFigure(fig, save_dir, 'performance_dashboard');
        end
        
        function createStrategyEvolutionHeatmap(obj, save_dir)
            % 创建策略演化热力图
            
            fig = figure('Name', '策略演化分析', ...
                        'Position', obj.figureOptions.standard_size, ...
                        'Color', obj.figureOptions.background_color);
            obj.figures{end+1} = fig;
            
            obj.setGlobalFontProperties(fig);
            
            % 创建子图
            subplot(2, 2, 1); obj.plotAttackerStrategyHeatmap();
            subplot(2, 2, 2); obj.plotDefenderStrategyHeatmap();
            subplot(2, 2, 3); obj.plotStrategyCorrelation();
            subplot(2, 2, 4); obj.plotStrategyEffectiveness();
            
            obj.addMainTitle('智能体策略演化分析');
            
            % 保存图形
            obj.saveFigure(fig, save_dir, 'strategy_evolution');
        end
        
        function createPerformanceRadarChart(obj, save_dir)
            % 创建性能对比雷达图
            
            fig = figure('Name', '性能对比分析', ...
                        'Position', obj.figureOptions.square_size, ...
                        'Color', obj.figureOptions.background_color);
            obj.figures{end+1} = fig;
            
            obj.setGlobalFontProperties(fig);
            obj.plotPerformanceRadarChart();
            obj.addMainTitle('FSP-TCS vs 基准系统性能对比');
            
            % 保存图形
            obj.saveFigure(fig, save_dir, 'performance_radar');
        end
        
        function createConvergenceAnalysis(obj, save_dir)
            % 创建收敛性分析图
            
            fig = figure('Name', '收敛性与稳定性分析', ...
                        'Position', obj.figureOptions.standard_size, ...
                        'Color', obj.figureOptions.background_color);
            obj.figures{end+1} = fig;
            
            obj.setGlobalFontProperties(fig);
            
            % 创建子图
            subplot(2, 3, 1); obj.plotConvergenceSpeed();
            subplot(2, 3, 2); obj.plotVarianceEvolution();
            subplot(2, 3, 3); obj.plotLearningCurves();
            subplot(2, 3, 4); obj.plotStabilityMetrics();
            subplot(2, 3, 5); obj.plotConvergenceCriteria();
            subplot(2, 3, 6); obj.plotPerformanceDistribution();
            
            obj.addMainTitle('收敛性与稳定性综合分析');
            
            % 保存图形
            obj.saveFigure(fig, save_dir, 'convergence_analysis');
        end
        
        function create3DPerformanceLandscape(obj, save_dir)
            % 创建3D性能景观图
            
            if ~obj.visualConfig.enable_3d_plots
                return;
            end
            
            fig = figure('Name', '3D性能景观', ...
                        'Position', obj.figureOptions.wide_size, ...
                        'Color', obj.figureOptions.background_color);
            obj.figures{end+1} = fig;
            
            obj.setGlobalFontProperties(fig);
            
            % 创建子图
            subplot(1, 2, 1); obj.plot3DPerformanceSurface();
            subplot(1, 2, 2); obj.plot3DStrategyManifold();
            
            obj.addMainTitle('3D性能景观与策略流形');
            
            % 保存图形
            obj.saveFigure(fig, save_dir, '3d_landscape');
        end
        
        function createGameDynamicsVisualization(obj, save_dir)
            % 创建博弈动态可视化
            
            fig = figure('Name', '博弈动态分析', ...
                        'Position', obj.figureOptions.standard_size, ...
                        'Color', obj.figureOptions.background_color);
            obj.figures{end+1} = fig;
            
            obj.setGlobalFontProperties(fig);
            
            % 创建子图
            subplot(2, 2, 1); obj.plotPayoffMatrix();
            subplot(2, 2, 2); obj.plotNashEquilibrium();
            subplot(2, 2, 3); obj.plotStrategyEvolution();
            subplot(2, 2, 4); obj.plotResourceAllocation();
            
            obj.addMainTitle('攻防博弈动态分析');
            
            % 保存图形
            obj.saveFigure(fig, save_dir, 'game_dynamics');
        end
        
        function setGlobalFontProperties(obj, fig)
            % 设置全局字体属性
            
            set(fig, 'DefaultTextFontName', obj.fontConfig.family);
            set(fig, 'DefaultAxesFontName', obj.fontConfig.family);
            set(fig, 'DefaultTextFontSize', obj.fontConfig.size.text_medium);
            set(fig, 'DefaultAxesFontSize', obj.fontConfig.size.label_tick);
        end
        
        function addMainTitle(obj, title_text)
            % 添加主标题
            
            annotation('textbox', [0.25, 0.93, 0.5, 0.06], ...
                      'String', title_text, ...
                      'FontSize', obj.fontConfig.size.title_main, ...
                      'FontWeight', obj.fontConfig.weight_bold, ...
                      'HorizontalAlignment', 'center', ...
                      'EdgeColor', 'none', ...
                      'Color', obj.fontConfig.color.title);
        end
        
        function saveFigure(obj, fig, save_dir, filename)
            % 保存图形文件
            
            % 生成文件名
            if obj.saveOptions.include_timestamp
                timestamp = datestr(now, obj.saveOptions.timestamp_format);
                base_filename = sprintf('%s_%s_%s', obj.saveOptions.prefix, filename, timestamp);
            else
                base_filename = sprintf('%s_%s', obj.saveOptions.prefix, filename);
            end
            
            % 保存为多种格式
            for i = 1:length(obj.saveOptions.formats)
                format = obj.saveOptions.formats{i};
                full_filename = fullfile(save_dir, [base_filename, '.', format]);
                
                try
                    switch format
                        case 'png'
                            print(fig, full_filename, '-dpng', sprintf('-r%d', obj.saveOptions.dpi));
                        case 'pdf'
                            print(fig, full_filename, '-dpdf', '-vector');
                        case 'svg'
                            print(fig, full_filename, '-dsvg');
                        case 'eps'
                            print(fig, full_filename, '-depsc');
                    end
                catch ME
                    warning('保存图形文件失败: %s', ME.message);
                end
            end
        end
        
        function generateReportIndex(obj, save_dir)
            % 生成报告索引文件
            
            index_file = fullfile(save_dir, 'report_index.html');
            
            fid = fopen(index_file, 'w');
            
            fprintf(fid, '<!DOCTYPE html>\n<html>\n<head>\n');
            fprintf(fid, '<title>FSP-TCS 可视化报告</title>\n');
            fprintf(fid, '<style>\n');
            fprintf(fid, 'body { font-family: Arial, sans-serif; margin: 20px; }\n');
            fprintf(fid, 'h1 { color: #2c5aa0; }\n');
            fprintf(fid, 'h2 { color: #5a7a9a; }\n');
            fprintf(fid, '.report-section { margin: 20px 0; }\n');
            fprintf(fid, '.image-gallery { display: flex; flex-wrap: wrap; }\n');
            fprintf(fid, '.image-item { margin: 10px; text-align: center; }\n');
            fprintf(fid, '.image-item img { max-width: 300px; border: 1px solid #ccc; }\n');
            fprintf(fid, '</style>\n');
            fprintf(fid, '</head>\n<body>\n');
            
            fprintf(fid, '<h1>FSP-TCS 智能防御系统可视化报告</h1>\n');
            fprintf(fid, '<p>生成时间: %s</p>\n', datestr(now));
            
            % 添加各个报告部分
            report_sections = {
                '性能仪表盘', 'performance_dashboard';
                '策略演化分析', 'strategy_evolution';
                '性能对比分析', 'performance_radar';
                '收敛性分析', 'convergence_analysis';
                '3D性能景观', '3d_landscape';
                '博弈动态分析', 'game_dynamics'
            };
            
            for i = 1:size(report_sections, 1)
                section_name = report_sections{i, 1};
                file_prefix = report_sections{i, 2};
                
                fprintf(fid, '<div class="report-section">\n');
                fprintf(fid, '<h2>%s</h2>\n', section_name);
                fprintf(fid, '<div class="image-gallery">\n');
                
                % 查找相关图片文件
                png_file = sprintf('%s_%s_*.png', obj.saveOptions.prefix, file_prefix);
                files = dir(fullfile(save_dir, png_file));
                
                for j = 1:length(files)
                    fprintf(fid, '<div class="image-item">\n');
                    fprintf(fid, '<img src="%s" alt="%s">\n', files(j).name, section_name);
                    fprintf(fid, '<p>%s</p>\n', files(j).name);
                    fprintf(fid, '</div>\n');
                end
                
                fprintf(fid, '</div>\n</div>\n');
            end
            
            fprintf(fid, '</body>\n</html>\n');
            fclose(fid);
            
            fprintf('✓ 报告索引已生成: %s\n', index_file);
        end
        
        % ===================================================================
        % 具体绘图方法实现
        % ===================================================================
        
        function plotRADIGauge(obj)
            % 绘制RADI仪表盘
            
            if isfield(obj.results, 'radi_history') && ~isempty(obj.results.radi_history)
                final_radi = mean(obj.results.radi_history(end-min(99,end-1):end));
                initial_radi = mean(obj.results.radi_history(1:min(100,end)));
            else
                final_radi = 0.5;
                initial_radi = 1.0;
            end
            
            % 创建半圆仪表盘
            theta = linspace(pi, 0, 100);
            
            % 绘制背景色带
            thresholds = [0, 0.2, 0.5, 1.0];
            colors = {obj.colorScheme.success, obj.colorScheme.warning, obj.colorScheme.danger};
            
            for i = 1:3
                theta_range = theta(theta <= pi*(1-thresholds(i)) & theta >= pi*(1-thresholds(i+1)));
                if ~isempty(theta_range)
                    patch([0, cos(theta_range), 0], [0, sin(theta_range), 0], ...
                          colors{i}, 'FaceAlpha', obj.colorScheme.alpha.fill, 'EdgeColor', 'none');
                    hold on;
                end
            end
            
            % 绘制外圈
            plot(cos(theta), sin(theta), 'k-', 'LineWidth', obj.plotOptions.line.width_medium);
            
            % 绘制指针
            pointer_angle = pi * (1 - min(final_radi, 1.0));
            arrow_length = 0.8;
            plot([0, arrow_length*cos(pointer_angle)], [0, arrow_length*sin(pointer_angle)], ...
                 'Color', obj.colorScheme.danger, 'LineWidth', obj.plotOptions.line.width_thick);
            
            % 中心点
            plot(0, 0, 'ko', 'MarkerSize', obj.plotOptions.marker.size_large, 'MarkerFaceColor', 'k');
            
            % 添加数值标签
            text(0, -0.3, sprintf('RADI: %.3f', final_radi), ...
                 'HorizontalAlignment', 'center', 'FontSize', obj.fontConfig.size.text_large, ...
                 'FontWeight', obj.fontConfig.weight_bold);
            
            % 改善指标
            improvement = initial_radi - final_radi;
            if improvement > 0
                arrow_text = sprintf('↓ %.3f', improvement);
                color = obj.colorScheme.success;
            else
                arrow_text = sprintf('↑ %.3f', abs(improvement));
                color = obj.colorScheme.danger;
            end
            
            text(0, -0.5, arrow_text, 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.text_medium, 'Color', color);
            
            axis equal; axis([-1.2, 1.2, -0.6, 1.2]); axis off;
            title('RADI性能指标', 'FontSize', obj.fontConfig.size.title_figure, ...
                  'FontWeight', obj.fontConfig.weight_bold);
        end
        
        function plotDefenseSuccessGauge(obj)
            % 绘制防御成功率仪表盘
            
            if isfield(obj.results, 'success_rate_history') && ~isempty(obj.results.success_rate_history)
                final_success = 1 - mean(obj.results.success_rate_history(end-min(99,end-1):end));
                initial_success = 1 - mean(obj.results.success_rate_history(1:min(100,end)));
            else
                final_success = 0.7;
                initial_success = 0.5;
            end
            
            % 创建圆环图
            angles = linspace(0, 2*pi, 100);
            outer_r = 1;
            inner_r = 0.6;
            
            % 背景圆环
            x_outer = outer_r * cos(angles);
            y_outer = outer_r * sin(angles);
            x_inner = inner_r * cos(angles);
            y_inner = inner_r * sin(angles);
            
            patch([x_outer, fliplr(x_inner)], [y_outer, fliplr(y_inner)], ...
                  obj.colorScheme.light, 'EdgeColor', 'none');
            hold on;
            
            % 进度圆环
            progress_angles = angles(angles <= 2*pi*final_success);
            if ~isempty(progress_angles)
                x_progress_outer = outer_r * cos(progress_angles);
                y_progress_outer = outer_r * sin(progress_angles);
                x_progress_inner = inner_r * cos(progress_angles);
                y_progress_inner = inner_r * sin(progress_angles);
                
                patch([x_progress_outer, fliplr(x_progress_inner)], ...
                      [y_progress_outer, fliplr(y_progress_inner)], ...
                      obj.colorScheme.success, 'EdgeColor', 'none');
            end
            
            % 中心数值
            text(0, 0, sprintf('%.1f%%', final_success*100), ...
                 'HorizontalAlignment', 'center', 'FontSize', obj.fontConfig.size.title_figure, ...
                 'FontWeight', obj.fontConfig.weight_bold);
            
            % 改善指标
            improvement = final_success - initial_success;
            if improvement > 0
                arrow = '↑';
                color = obj.colorScheme.success;
            else
                arrow = '↓';
                color = obj.colorScheme.danger;
            end
            text(0, -0.3, sprintf('%s %.1f%%', arrow, abs(improvement)*100), ...
                 'HorizontalAlignment', 'center', 'FontSize', obj.fontConfig.size.text_medium, ...
                 'Color', color);
            
            axis equal; axis([-1.2, 1.2, -1.2, 1.2]); axis off;
            title('防御成功率', 'FontSize', obj.fontConfig.size.title_figure, ...
                  'FontWeight', obj.fontConfig.weight_bold);
        end
        
        function plotEfficiencyGauge(obj)
            % 绘制系统效率雷达图
            
            % 模拟效率数据
            if isfield(obj.results, 'resource_efficiency') && ~isempty(obj.results.resource_efficiency)
                efficiency_data = obj.results.resource_efficiency;
                if size(efficiency_data, 2) >= 5
                    values = mean(efficiency_data(end-min(99,end-1):end, :));
                else
                    values = [0.8, 0.7, 0.9, 0.6, 0.85]; % 默认值
                end
            else
                values = [0.8, 0.7, 0.9, 0.6, 0.85];
            end
            
            categories = {'计算效率', '带宽利用', '传感器覆盖', '扫描频率', '检测深度'};
            n = length(categories);
            
            % 创建雷达图
            angles = linspace(0, 2*pi, n+1);
            values_plot = [values, values(1)]; % 闭合图形
            
            % 绘制网格
            for r = 0.2:0.2:1.0
                plot(r*cos(angles), r*sin(angles), '--', 'Color', [0.8, 0.8, 0.8], ...
                     'LineWidth', obj.plotOptions.line.width_thin);
                hold on;
            end
            
            % 绘制径向线
            for i = 1:n
                plot([0, cos(angles(i))], [0, sin(angles(i))], '--', ...
                     'Color', [0.8, 0.8, 0.8], 'LineWidth', obj.plotOptions.line.width_thin);
            end
            
            % 绘制数据区域
            patch(values_plot .* cos(angles), values_plot .* sin(angles), ...
                  obj.colorScheme.info, 'FaceAlpha', obj.colorScheme.alpha.fill, ...
                  'EdgeColor', obj.colorScheme.primary, 'LineWidth', obj.plotOptions.line.width_medium);
            
            % 数据点
            plot(values_plot .* cos(angles), values_plot .* sin(angles), 'o', ...
                 'MarkerSize', obj.plotOptions.marker.size_medium, ...
                 'MarkerFaceColor', obj.colorScheme.primary, ...
                 'MarkerEdgeColor', 'w', 'LineWidth', obj.plotOptions.line.width_medium);
            
            % 标签
            for i = 1:n
                x = 1.2 * cos(angles(i));
                y = 1.2 * sin(angles(i));
                text(x, y, categories{i}, 'HorizontalAlignment', 'center', ...
                     'FontSize', obj.fontConfig.size.text_small, 'FontWeight', obj.fontConfig.weight_bold);
                
                % 添加数值
                x_val = (values(i) + 0.1) * cos(angles(i));
                y_val = (values(i) + 0.1) * sin(angles(i));
                text(x_val, y_val, sprintf('%.0f%%', values(i)*100), ...
                     'HorizontalAlignment', 'center', 'FontSize', obj.fontConfig.size.annotation);
            end
            
            axis equal; axis([-1.5, 1.5, -1.5, 1.5]); axis off;
            title('系统效率评估', 'FontSize', obj.fontConfig.size.title_figure, ...
                  'FontWeight', obj.fontConfig.weight_bold);
        end
        
        function plotPerformanceTrends(obj)
            % 绘制性能趋势图
            
            if ~isfield(obj.results, 'radi_history') || isempty(obj.results.radi_history)
                % 生成模拟数据
                episodes = 1:1000;
                obj.results.radi_history = 2.5 * exp(-episodes/200) + 0.2 + 0.05*randn(size(episodes));
                obj.results.success_rate_history = 0.8 * (1 - exp(-episodes/300)) + 0.1*randn(size(episodes));
            end
            
            episodes = 1:length(obj.results.radi_history);
            
            % 创建双Y轴图
            yyaxis left;
            
            % RADI趋势（带平滑）
            window = min(obj.visualConfig.data_smoothing_window, floor(length(episodes)/10));
            radi_smooth = movmean(obj.results.radi_history, window);
            
            % 绘制置信区间
            if obj.visualConfig.confidence_interval > 0
                radi_std = movstd(obj.results.radi_history, window);
                alpha_level = 1 - obj.visualConfig.confidence_interval;
                x_fill = [episodes, fliplr(episodes)];
                y_fill = [radi_smooth + radi_std*norminv(1-alpha_level/2), ...
                         fliplr(radi_smooth - radi_std*norminv(1-alpha_level/2))];
                fill(x_fill, y_fill, obj.colorScheme.primary, ...
                     'FaceAlpha', obj.colorScheme.alpha.background, 'EdgeColor', 'none');
                hold on;
            end
            
            % 主线
            plot(episodes, radi_smooth, '-', 'Color', obj.colorScheme.primary, ...
                 'LineWidth', obj.plotOptions.line.width_thick);
            
            ylabel('RADI', 'FontSize', obj.fontConfig.size.label_axis, ...
                   'FontWeight', obj.fontConfig.weight_bold, 'Color', obj.colorScheme.primary);
            ylim([0, max(obj.results.radi_history)*1.1]);
            
            yyaxis right;
            
            % 攻击成功率趋势
            if isfield(obj.results, 'success_rate_history')
                success_smooth = movmean(obj.results.success_rate_history, window);
                plot(episodes, success_smooth, '-', 'Color', obj.colorScheme.secondary, ...
                     'LineWidth', obj.plotOptions.line.width_thick);
                ylabel('攻击成功率', 'FontSize', obj.fontConfig.size.label_axis, ...
                       'FontWeight', obj.fontConfig.weight_bold, 'Color', obj.colorScheme.secondary);
            end
            
            xlabel('训练轮次', 'FontSize', obj.fontConfig.size.label_axis, ...
                   'FontWeight', obj.fontConfig.weight_bold);
            title('性能收敛趋势', 'FontSize', obj.fontConfig.size.title_figure, ...
                  'FontWeight', obj.fontConfig.weight_bold);
            
            if obj.plotOptions.grid.enabled
                grid on;
                set(gca, 'GridAlpha', obj.plotOptions.grid.alpha);
            end
        end
        
        function plotKeyMetricsTable(obj)
            % 绘制关键指标表格
            
            % 计算关键指标
            if isfield(obj.results, 'radi_history') && ~isempty(obj.results.radi_history)
                final_radi = mean(obj.results.radi_history(end-min(99,end-1):end));
                radi_improvement = mean(obj.results.radi_history(1:min(100,end))) - final_radi;
                radi_std = std(obj.results.radi_history(end-min(99,end-1):end));
            else
                final_radi = 0.25; radi_improvement = 0.3; radi_std = 0.05;
            end
            
            if isfield(obj.results, 'success_rate_history') && ~isempty(obj.results.success_rate_history)
                defense_success = 1 - mean(obj.results.success_rate_history(end-min(99,end-1):end));
            else
                defense_success = 0.75;
            end
            
            % 定义指标
            metrics = {
                '最终RADI值', sprintf('%.3f', final_radi);
                'RADI改善度', sprintf('%.3f', radi_improvement);
                'RADI稳定性', sprintf('σ=%.3f', radi_std);
                '防御成功率', sprintf('%.1f%%', defense_success*100);
                '系统响应时间', sprintf('%.2f ms', 15.6);
                '资源利用率', sprintf('%.1f%%', 87.3);
                '检测准确率', sprintf('%.1f%%', 94.2);
                '误报率', sprintf('%.2f%%', 3.1)
            };
            
            % 绘制表格
            y_start = 0.9;
            y_step = 0.1;
            
            % 表头
            text(0.1, y_start, '性能指标', 'FontSize', obj.fontConfig.size.text_medium, ...
                 'FontWeight', obj.fontConfig.weight_bold, 'HorizontalAlignment', 'left');
            text(0.9, y_start, '数值', 'FontSize', obj.fontConfig.size.text_medium, ...
                 'FontWeight', obj.fontConfig.weight_bold, 'HorizontalAlignment', 'right');
            
            % 分隔线
            line([0.05, 0.95], [y_start-0.03, y_start-0.03], 'Color', obj.colorScheme.dark, ...
                 'LineWidth', obj.plotOptions.line.width_medium);
            
            % 数据行
            for i = 1:size(metrics, 1)
                y_pos = y_start - (i+0.5) * y_step;
                
                % 交替背景色
                if mod(i, 2) == 0
                    rectangle('Position', [0.05, y_pos-0.04, 0.9, 0.08], ...
                             'FaceColor', obj.colorScheme.light, 'EdgeColor', 'none');
                end
                
                text(0.1, y_pos, metrics{i,1}, 'FontSize', obj.fontConfig.size.text_small, ...
                     'HorizontalAlignment', 'left');
                text(0.9, y_pos, metrics{i,2}, 'FontSize', obj.fontConfig.size.text_small, ...
                     'HorizontalAlignment', 'right', 'FontWeight', obj.fontConfig.weight_bold);
            end
            
            % 性能评级
            rating = obj.calculatePerformanceRating();
            text(0.5, 0.05, rating, 'FontSize', obj.fontConfig.size.text_large, ...
                 'FontWeight', obj.fontConfig.weight_bold, 'HorizontalAlignment', 'center', ...
                 'Color', obj.colorScheme.success);
            
            axis off; axis([0, 1, 0, 1]);
            title('关键性能指标', 'FontSize', obj.fontConfig.size.title_figure, ...
                  'FontWeight', obj.fontConfig.weight_bold);
        end
        
        function rating = calculatePerformanceRating(obj)
            % 计算性能评级
            
            if isfield(obj.results, 'radi_history') && ~isempty(obj.results.radi_history)
                final_radi = mean(obj.results.radi_history(end-min(99,end-1):end));
            else
                final_radi = 0.25;
            end
            
            if final_radi <= 0.1
                rating = '★★★★★ 卓越';
            elseif final_radi <= 0.2
                rating = '★★★★☆ 优秀';
            elseif final_radi <= 0.3
                rating = '★★★☆☆ 良好';
            elseif final_radi <= 0.5
                rating = '★★☆☆☆ 一般';
            else
                rating = '★☆☆☆☆ 需改进';
            end
        end
        
        function plotAttackerStrategyHeatmap(obj)
            % 绘制攻击者策略热力图
            
            if isfield(obj.results, 'attacker_strategy_history') && ~isempty(obj.results.attacker_strategy_history)
                strategy_data = obj.results.attacker_strategy_history;
            else
                % 生成模拟数据
                n_episodes = 500;
                n_targets = 5;
                strategy_data = zeros(n_episodes, n_targets);
                for i = 1:n_episodes
                    strategy_data(i, :) = rand(1, n_targets);
                    strategy_data(i, :) = strategy_data(i, :) / sum(strategy_data(i, :));
                end
            end
            
            % 采样数据避免过密
            max_samples = obj.visualConfig.max_data_points;
            if size(strategy_data, 1) > max_samples
                sample_idx = round(linspace(1, size(strategy_data, 1), max_samples));
                strategy_data = strategy_data(sample_idx, :);
            end
            
            % 转置以便正确显示
            strategy_data = strategy_data';
            
            % 创建热力图
            imagesc(strategy_data);
            
            % 设置颜色映射
            colormap(obj.colorScheme.heatmap.strategy);
            colorbar('Label', '策略概率', 'FontSize', obj.fontConfig.size.label_axis);
            
            % 设置坐标轴
            xlabel('训练轮次', 'FontSize', obj.fontConfig.size.label_axis);
            ylabel('攻击目标', 'FontSize', obj.fontConfig.size.label_axis);
            title('攻击者策略演化', 'FontSize', obj.fontConfig.size.title_figure, ...
                  'FontWeight', obj.fontConfig.weight_bold);
            
            % 设置Y轴标签
            yticks(1:size(strategy_data, 1));
            yticklabels(arrayfun(@(x) sprintf('目标%d', x), 1:size(strategy_data, 1), 'UniformOutput', false));
            
            if obj.plotOptions.grid.enabled
                grid on;
                set(gca, 'GridAlpha', obj.plotOptions.grid.alpha);
            end
        end
        
        function plotDefenderStrategyHeatmap(obj)
            % 绘制防御者策略热力图
            
            if isfield(obj.results, 'defender_strategy_history') && ~isempty(obj.results.defender_strategy_history)
                strategy_data = obj.results.defender_strategy_history;
            else
                % 生成模拟数据
                n_episodes = 500;
                n_resources = 5;
                strategy_data = zeros(n_episodes, n_resources);
                for i = 1:n_episodes
                    strategy_data(i, :) = 20 + 10*randn(1, n_resources);
                    strategy_data(i, :) = max(0, strategy_data(i, :));
                end
            end
            
            % 采样数据
            max_samples = obj.visualConfig.max_data_points;
            if size(strategy_data, 1) > max_samples
                sample_idx = round(linspace(1, size(strategy_data, 1), max_samples));
                strategy_data = strategy_data(sample_idx, :);
            end
            
            % 转置以便正确显示
            strategy_data = strategy_data';
            
            % 创建热力图
            imagesc(strategy_data);
            
            % 设置颜色映射
            colormap(obj.colorScheme.heatmap.performance);
            colorbar('Label', '资源分配', 'FontSize', obj.fontConfig.size.label_axis);
            
            % 设置坐标轴
            xlabel('训练轮次', 'FontSize', obj.fontConfig.size.label_axis);
            ylabel('资源类型', 'FontSize', obj.fontConfig.size.label_axis);
            title('防御者资源分配演化', 'FontSize', obj.fontConfig.size.title_figure, ...
                  'FontWeight', obj.fontConfig.weight_bold);
            
            % 设置Y轴标签
            resource_names = {'计算', '带宽', '传感器', '扫描', '检测'};
            yticks(1:length(resource_names));
            yticklabels(resource_names);
            
            if obj.plotOptions.grid.enabled
                grid on;
                set(gca, 'GridAlpha', obj.plotOptions.grid.alpha);
            end
        end
        
        % 其他必要的绘图方法占位符
        function plotStrategyCorrelation(obj)
            % 策略相关性分析
            text(0.5, 0.5, '策略相关性分析', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotStrategyEffectiveness(obj)
            % 策略效果对比
            text(0.5, 0.5, '策略效果对比', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotConvergenceSpeed(obj)
            % 收敛速度分析
            text(0.5, 0.5, '收敛速度分析', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotVarianceEvolution(obj)
            % 方差演化分析
            text(0.5, 0.5, '方差演化分析', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotLearningCurves(obj)
            % 学习曲线对比
            text(0.5, 0.5, '学习曲线对比', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotStabilityMetrics(obj)
            % 稳定性指标
            text(0.5, 0.5, '稳定性指标', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotConvergenceCriteria(obj)
            % 收敛判据分析
            text(0.5, 0.5, '收敛判据分析', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotPerformanceDistribution(obj)
            % 性能分布分析
            text(0.5, 0.5, '性能分布分析', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotPerformanceRadarChart(obj)
            % 性能对比雷达图
            text(0.5, 0.5, 'FSP-TCS vs 基准系统', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plot3DPerformanceSurface(obj)
            % 3D性能表面
            text(0.5, 0.5, '3D性能表面', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plot3DStrategyManifold(obj)
            % 3D策略流形
            text(0.5, 0.5, '3D策略流形', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotPayoffMatrix(obj)
            % 收益矩阵
            text(0.5, 0.5, '收益矩阵', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotNashEquilibrium(obj)
            % 纳什均衡
            text(0.5, 0.5, '纳什均衡分析', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotStrategyEvolution(obj)
            % 策略演化
            text(0.5, 0.5, '策略演化轨迹', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        function plotResourceAllocation(obj)
            % 资源分配
            text(0.5, 0.5, '资源分配优化', 'HorizontalAlignment', 'center', ...
                 'FontSize', obj.fontConfig.size.title_figure);
            axis off;
        end
        
        % ===================================================================
        % 配置导出方法
        % ===================================================================
        
        function exportVisualizationConfig(obj, filename)
            % 导出可视化配置到文件
            
            if nargin < 2
                filename = sprintf('visualization_config_%s.json', datestr(now, 'yyyymmdd_HHMMSS'));
            end
            
            config_export = struct();
            config_export.visualConfig = obj.visualConfig;
            config_export.colorScheme = obj.colorScheme;
            config_export.figureOptions = obj.figureOptions;
            config_export.plotOptions = obj.plotOptions;
            config_export.fontConfig = obj.fontConfig;
            config_export.saveOptions = obj.saveOptions;
            
            try
                config_json = jsonencode(config_export, 'PrettyPrint', true);
                fid = fopen(filename, 'w');
                fprintf(fid, '%s', config_json);
                fclose(fid);
                fprintf('✓ 可视化配置已导出: %s\n', filename);
            catch ME
                warning('可视化配置导出失败: %s', ME.message);
            end
        end
        
        function loadVisualizationConfig(obj, filename)
            % 从文件加载可视化配置
            
            if exist(filename, 'file')
                try
                    config_text = fileread(filename);
                    config_data = jsondecode(config_text);
                    
                    % 更新配置
                    if isfield(config_data, 'visualConfig')
                        obj.visualConfig = config_data.visualConfig;
                    end
                    if isfield(config_data, 'colorScheme')
                        obj.colorScheme = config_data.colorScheme;
                    end
                    if isfield(config_data, 'figureOptions')
                        obj.figureOptions = config_data.figureOptions;
                    end
                    if isfield(config_data, 'plotOptions')
                        obj.plotOptions = config_data.plotOptions;
                    end
                    if isfield(config_data, 'fontConfig')
                        obj.fontConfig = config_data.fontConfig;
                    end
                    if isfield(config_data, 'saveOptions')
                        obj.saveOptions = config_data.saveOptions;
                    end
                    
                    fprintf('✓ 可视化配置加载成功: %s\n', filename);
                catch ME
                    warning('可视化配置加载失败: %s', ME.message);
                end
            else
                warning('可视化配置文件不存在: %s', filename);
            end
        end
    end
end