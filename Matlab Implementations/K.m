% Input: vector x of d rows, 1 column
% vector xp of d rows, 1 column
% Output: kernel K(x,xp) = exp(-1/2 * norm(x-xp)^2)
% Example on how to call the function: v = K([1; 4; 3],[2; 5; -1]);
function v = K(x,xp)
v = exp(-1/2 * sum((x-xp).^2));
