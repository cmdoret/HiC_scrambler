from collections import OrderedDict

def lempel_complexity(seq : str):
    
    words = OrderedDict()
    size_seq = len(seq)

    ind_seq= 0
    ind_end_word = 1

    while ind_end_word <= size_seq:

        
        while seq[ind_seq:ind_end_word] in words and ind_end_word <= size_seq:
            
            ind_end_word +=1
        words[seq[ind_seq:ind_end_word]] = 0
        ind_seq = ind_end_word
        ind_end_word = ind_seq+1
    return len(words)/size_seq