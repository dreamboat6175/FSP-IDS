%% ParallelComputing.m - 并行计算支持类
classdef ParallelComputing
    methods (Static)
        function setup()
            % 设置并行计算环境
            if license('test', 'Distrib_Computing_Toolbox')
                pool = gcp('nocreate');
                if isempty(pool)
                    % 获取可用核心数
                    num_cores = feature('numcores');
                    % 使用核心数-1，保留一个核心给系统
                    num_workers = max(1, num_cores - 1);
                    
                    try
                        parpool('local', num_workers);
                        fprintf('✓ 并行计算池已启动，使用 %d 个工作进程\n', num_workers);
                    catch ME
                        warning('无法启动并行池: %s', ME.message);
                        fprintf('将使用单线程模式\n');
                    end
                else
                    fprintf('✓ 并行计算池已存在，使用 %d 个工作进程\n', pool.NumWorkers);
                end
            else
                fprintf('未检测到并行计算工具箱，使用单线程模式\n');
            end
        end
        
        function cleanup()
            % 清理并行计算资源
            pool = gcp('nocreate');
            if ~isempty(pool)
                delete(pool);
                fprintf('并行计算池已关闭\n');
            end
        end
        
        function result = parallelRun(func, data, options)
            % 并行执行函数
            if nargin < 3
                options = struct();
            end
            
            pool = gcp('nocreate');
            if ~isempty(pool) && pool.NumWorkers > 1
                % 使用并行计算
                result = cell(length(data), 1);
                parfor i = 1:length(data)
                    result{i} = func(data{i});
                end
            else
                % 串行计算
                result = cell(length(data), 1);
                for i = 1:length(data)
                    result{i} = func(data{i});
                end
            end
        end
    end
end
