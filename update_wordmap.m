function [wordMap] = update_wordmap(filename, wordMapExist)

headLine = true;
separater = '::';

words = [];

fid = fopen(filename, 'r');
line = fgets(fid);

ind = 1;
while ischar(line)
    if headLine
        line = fgets(fid);
        headLine = false;
    end
    attrs = strsplit(line, separater);    
    s = attrs{2};
    w = strsplit(s);
    
    for eachword=w
        if isKey(wordMapExist, eachword) == false
            words = [words eachword];
        end
    end
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end
words = unique(words);

length_wordmap = length(wordMapExist);
wordMap = wordMapExist;

count = 1;
for word = words
    wordMap(word{1}) = count + length_wordmap;
    count = count + 1;
end