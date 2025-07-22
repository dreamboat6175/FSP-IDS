function safe_rewards = safeAssignRewards(reward_data, expected_cols, episode_idx, data_type)
%SAFEASSIGNREWARDS 安全地分配奖励数据，确保维度匹配
%
% 输入参数:
%   reward_data - 奖励数据（标量、向量或矩阵）
%   expected_cols - 期望的列数（通常是防御者数量）
%   episode_idx - 当前episode索引（用于调试）
%   data_type - 数据类型描述（用于调试输出）
%
% 输出参数:
%   safe_rewards - 安全的奖励向量 [1 x expected_cols]

try
    % 输入验证
    if nargin < 3
        episode_idx = 0;
    end
    if nargin < 4
        data_type = 'unknown';
    end
    
    % 处理空数据
    if isempty(reward_data)
        safe_rewards = zeros(1, expected_cols);
        return;
    end
    
    % 确保reward_data是数值类型
    if ~isnumeric(reward_data)
        warning('safeAssignRewards:InvalidType', '奖励数据不是数值类型，使用零值');
        safe_rewards = zeros(1, expected_cols);
        return;
    end
    
    % 处理NaN和Inf值
    reward_data(isnan(reward_data)) = 0;
    reward_data(isinf(reward_data)) = 0;
    
    % 转换为行向量
    reward_data = reward_data(:)';
    
    % 根据输入数据长度进行处理
    if length(reward_data) == expected_cols
        % 长度完全匹配
        safe_rewards = reward_data;
    elseif length(reward_data) == 1
        % 标量输入：复制到所有列
        safe_rewards = repmat(reward_data, 1, expected_cols);
    elseif length(reward_data) < expected_cols
        % 输入长度不足：填充零值
        padding_length = expected_cols - length(reward_data);
        safe_rewards = [reward_data, zeros(1, padding_length)];
        
        if episode_idx > 0
            fprintf('[DEBUG] safeAssignRewards: %s 在episode %d 维度不足，已填充零值\n', ...
                    data_type, episode_idx);
        end
    else
        % 输入长度过多：截断到预期长度
        safe_rewards = reward_data(1:expected_cols);
        
        if episode_idx > 0
            fprintf('[DEBUG] safeAssignRewards: %s 在episode %d 维度过多，已截断\n', ...
                    data_type, episode_idx);
        end
    end
    
    % 最终验证
    if length(safe_rewards) ~= expected_cols
        warning('safeAssignRewards:DimensionError', ...
                '无法修复维度不匹配问题，使用零向量 (期望: %d, 实际: %d)', ...
                expected_cols, length(safe_rewards));
        safe_rewards = zeros(1, expected_cols);
    end
    
    % 确保输出是行向量
    safe_rewards = safe_rewards(:)';
    
catch ME
    % 错误处理：返回零向量
    warning('safeAssignRewards:Error', '奖励分配出错: %s，使用零向量', ME.message);
    safe_rewards = zeros(1, expected_cols);
end

end