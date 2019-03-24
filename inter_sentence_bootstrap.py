import spacy
nlp=spacy.load("en")
#input indx -> int
#input window_size -> int
#input length -> int
#output list[start_index,end_index]
def align_window(indx,window_size,length):
    if window_size>=length:
        return[0,length]
    else:
        l=r=window_size//2
        if indx<l:
            r=r+(l-indx)+indx
            #l=0
            return [0,r+1]
        elif length-(indx+1)<r:
            l=l+(r-(length-(indx+1)))
            l=indx-l
            return [l,length]
        else:
            return[indx-l,indx+r+1]
#input sent ->String
#input tag-> 1d list
#input window_size-> int 
#input stop_words->False
#ouput nested list
def flex_window_patEx(sent,tag_list,window_size=5,stop_words=False,return_type="dep"):
    tptfrm=triples(sent,stop_words)
    word=tptfrm[0]
    dep=tptfrm[1]
    tag=tptfrm[2]
    retdict={}
    for i in tag_list:
        for j,k in enumerate(dep):
            if k==i and tag[j]!='CD':
                if i not in retdict:
                    ind=align_window(j,window_size,len(dep))
                    if return_type=="dep":
                        retdict[i]=[dep[ind[0]:ind[1]]]
                    else:
                        retdict[i]=[tag[ind[0]:ind[1]]]
                else:
                    ind=align_window(j,window_size,len(dep))
                    if return_type=="dep":
                        retdict[i].append(dep[ind[0]:ind[1]])
                    else:
                        retdict[i].append(tag[ind[0]:ind[1]])
    return retdict
#input sent as string
#input s_words as boolean
#output list(word,dep,pos tag)
def triples(sent,s_words=False):
    from nltk.corpus import stopwords
    sw=stopwords.words('english')
    doc=nlp(sent)
    word=[]
    dep=[]
    tag=[]
    if s_words:
        for i in doc:
            if i.lower_ not in sw and i.dep_!="punct":
                word.append(i.text)
                dep.append(i.dep_)
                tag.append(i.tag_)
    else:
        for i in doc:
            word.append(i.text)
            dep.append(i.dep_)
            tag.append(i.tag_)
    return [word,dep,tag]
#input sent_obj->spacy obj as spacy obj
#input s_words as boolean
#output list(word,dep,pos tag)
def opt_triples(sent_obj,s_words=False):
    from nltk.corpus import stopwords
    sw=stopwords.words('english')
    word=[]
    dep=[]
    tag=[]
    if s_words:
        for i in sent_obj:
            if i.lower_ not in sw and i.dep_!="punct":
                word.append(i.text)
                dep.append(i.dep_)
                tag.append(i.tag_)
    else:
        for i in sent_obj:
            word.append(i.text)
            dep.append(i.dep_)
            tag.append(i.tag_)
    return [word,dep,tag]
#input s->1d list or string
#input t->1d list or string
#output jaro distance in float
def jaro(s, t):
    s_len = len(s)
    t_len = len(t)
 
    if s_len == 0 and t_len == 0:
        return 1
 
    match_distance = (max(s_len, t_len) // 2) - 1
 
    s_matches = [False] * s_len
    t_matches = [False] * t_len
 
    matches = 0
    transpositions = 0
 
    for i in range(s_len):
        start = max(0, i-match_distance)
        end = min(i+match_distance+1, t_len)
 
        for j in range(start, end):
            if t_matches[j]:
                continue
            if s[i] != t[j]:
                continue
            s_matches[i] = True
            t_matches[j] = True
            matches += 1
            break
 
    if matches == 0:
        return 0
 
    k = 0
    for i in range(s_len):
        if not s_matches[i]:
            continue
        while not t_matches[k]:
            k += 1
        if s[i] != t[k]:
            transpositions += 1
        k += 1
 
    return ((matches / s_len) +(matches / t_len) +((matches - transpositions/2) / matches)) / 3
#input ref_pat->list
#input eval_pat->list
#output result->list
def tag_inter_jaro(ref_pat,eval_pat):
    result=[{"sent1":[]},{"sent2":[]}]
    if {} in ref_pat:
        raise ValueError("Please check the referrence pattern:"+str(ref_pat))
    for i,j in enumerate(ref_pat):
        for ky,vl in j.items():
            if ky not in eval_pat[i]:
                result[i]["sent"+str(i+1)].append([ky,vl[0],0.0])
            else:
                maxj=0.0
                jscl=[]
                for m in vl:
                    for n in eval_pat[i][ky]:
                        jsc=jaro(m,n)
                        if maxj<=jsc:
                            maxj=jsc
                            jscl=[ky,n,jsc]
                result[i]["sent"+str(i+1)].append(jscl)
    return result
#input pair_sent->list of strings or sentence of len==2
#input ref_pat->list
#input tags->list
#input window_len->int
#input stop_words->boolean
#output list
def tag_inter_pattern_score(pair_sent,ref_pat,tags=["nsubj","dobj"],window_len=6,stop_words=True):
    sent_list=[]
    for i in pair_sent:
        sent_list.append(flex_window_patEx(i,tags,window_len,stop_words))
    return inter_jaro(ref_pat,sent_list)
#input result_dict->dict dictionary from pickle
#output tot_pat->dict
def inter_pattern_conv(result_dict):
    tot_pat={}
    for i in result_dict:
        for j in result_dict[i]:
            if i not in tot_pat:
                tot_pat[i]=inter_seed_pattern(j)
            else:
                tot_pat[i].extend(inter_seed_pattern(j))
    return tot_pat
#input sent_list->list of strings
#input windw_len->int
#input stop_words->Boolean
#input tagid->string
#output final_patterns->list list of patterns for each string
def inter_seed_pattern(sent_list,window_len=6,stop_words=True,tagid="dep"):
    ret_list=[]
    sent1_pat=[]
    sent2_pat=[]
    final_patterns=[]
    for i,j in enumerate(sent_list):
        tptfrm=triples(j,stop_words)
        word=tptfrm[0]
        dep=tptfrm[1]
        tag=tptfrm[2]
        maxt=0.0
        pat=[]
        if tagid=="dep":
            res=[dep[k:k+window_len] for k in range(len(dep)-window_len+1)]
        else:
            res=[tag[k:k+window_len] for k in range(len(tag)-window_len+1)]
        if i==0:
            sent1_pat=res
        elif i==1:
            sent2_pat=res
        else:
            raise ValueError("For this length of the input sentence list must be equal to 2")
    for i in sent1_pat:
        for j in sent2_pat:
            final_patterns.append([i,j])
    return final_patterns
#input ip_list->2d neseted list
def frequent_patterns(ip_list):
    import numpy as np
    ls={}
    for i in ip_list:
        for j in i:
            j=",".join(j)
            if j not in ls:
                ls[j]=[j.split(","),1]
            else:
                ls[j][1]=ls[j][1]+1
    f=[j[1] for i,j in ls.items()]
    f_i=np.argsort(f)
    f_i=f_i[::-1]
    f_l=list(ls)
    freq=[ls[f_l[i]] for i in f_i]
    return freq
#input seed_pat->dict
#input filter_val->float shoulld always between(0.0 to 1.0)
#output fil_seed_scr->dict filtered dictionary
def filter_seed_pattern(seed_scr,filter_val=.60):
    fil_seed_scr={}
    for i in seed_scr:
        fil_seed_scr[i]=[j  for j in seed_scr[i] if j[2]<filter_val]
    return fil_seed_scr
#input sent_list->list of strings or sentence of len==2
#input ref_pat->list
#input window_len->int
#input stop_words->boolean
#input tagid->sring
#output list
def inter_sent_pattern(sent_list,ref_pat,window_len=6,stop_words=True,tagid="dep"):
    if ref_pat==[] or sent_list==[]:
        raise ValueError("Please enter a valid pattern")
    if len(sent_list)!=len(ref_pat):
        raise ValueError("Length of the list sentence:",len(sent_list)," and length of the reference pattern:",len(ref_pat) ,"must be same")
    return_list=[]
    for i,j in enumerate(sent_list):
        tptfrm=triples(j,stop_words)
        word=tptfrm[0]
        dep=tptfrm[1]
        tag=tptfrm[2]
        maxt=0.0
        pat=[]
        if tagid=="dep":
            res=[dep[k:k+window_len] for k in range(len(dep)-window_len+1)]
            for m in res:
                sc=jaro(m,ref_pat[i])
                if sc>=maxt:
                    maxt=sc
                    pat=m
        else:
            res=[tag[k:k+window_len] for k in range(len(tag)-window_len+1)]
            for m in res:
                sc=jaro(m,ref_pat[i])
                if sc>=maxt:
                    maxt=sc
                    pat=m
        return_list.append([pat,maxt])
    return return_list
#input sent_obj_list->list of  spacy doc token of len==2
#input ref_pat->list
#input window_len->int
#input stop_words->boolean
#input tagid->sring
#output list
def opt_inter_sent_pattern(sent_obj_list,ref_pat,window_len=6,stop_words=True,tagid="dep"):
    if ref_pat==[] or sent_obj_list==[]:
        raise ValueError("Please enter a valid pattern")
    if len(sent_obj_list)!=len(ref_pat):
        raise ValueError("Length of the list sentence:",len(sent_obj_list)," and length of the reference pattern:",len(ref_pat) ,"must be same")
    return_list=[]
    for i,j in enumerate(sent_obj_list):
        tptfrm=opt_triples(j,stop_words)
        word=tptfrm[0]
        dep=tptfrm[1]
        tag=tptfrm[2]
        maxt=0.0
        pat=[]
        if tagid=="dep":
            res=[dep[k:k+window_len] for k in range(len(dep)-window_len+1)]
            for m in res:
                sc=jaro(m,ref_pat[i])
                if sc>=maxt:
                    maxt=sc
                    pat=m
        else:
            res=[tag[k:k+window_len] for k in range(len(tag)-window_len+1)]
            for m in res:
                sc=jaro(m,ref_pat[i])
                if sc>=maxt:
                    maxt=sc
                    pat=m
        return_list.append([pat,maxt])
    return return_list
#input seed_dict->dict with seed patterns
#output result_dict->dict element counted seperately for sent1 and sent2 group
def count_elements(seed_dict):
    from collections import Counter
    result_dict=dict()
    for kt in seed_dict:
        k1=[i[0] for i in seed_dict[kt]]
        k11=[]
        for i in k1:
            if i not in k11:
                k11.append(i)
        ka=[j for i in k11 for j in i]
        c1=Counter(ka)
        result_dict[kt]={}
        result_dict[kt]['sent1']=dict(c1)
        k2=[i[1] for i in seed_dict[kt]]
        k22=[]
        for i in k2:
            if i not in k22:
                k22.append(i)
        kb=[j for i in k22 for j in i]
        c2=Counter(kb)
        result_dict[kt]['sent2']=dict(c2)
    return result_dict
#input seed_dict->dict dict from seed dict
#output split_seed->dict splited dict
def split_seed_dict(seed_dict):
    split_seed={}
    for i in seed_dict:
        sent1=[j[0] for j in seed_dict[i]]
        s1=[]
        for k in sent1:
            if k not in s1:
                s1.append(k)
        sent2=[j[1] for j in seed_dict[i]]
        s2=[]
        for k in sent2:
            if k not in s2:
                s2.append(k)
        split_seed[i]={'sent1':s1}
        split_seed[i]['sent2']=s2
    return split_seed
#input file_name->string input file name
#input ref_pat->list input inter pattern
#input label-> string refer dictionary keys for label
#input window_size->int By default is 6
#input stop_words->boolean By default is True
#input tagid->string By default is dep 
#output dictionary
def inter_sent_pattern_tagger(file_name,ref_pat,label,window_size=6,stop_words=True,tagid="dep"):
    import os
    import pickle
    tags=["<arguments>","<identify>","<facts>","<decision>","<ratio>"]
    print("@"+file_name+"\n")
    pick_fl="/home/judson/Desktop/sentenceSeg/clean_sent_spacy/"+file_name.split("/")[-1].split(".")[0]+".pickle"
    pat_collect=[]
    new_list=[]
    tagged_sent=[]
    flag=0
    sent_list=open(file_name,"r").readlines()
    pickle_list=pickle.load(open(pick_fl,"rb"))
    if len(sent_list)!=len(pickle_list):
        return "check the files: "+file_name+", "+pick_fl
    result_list=[[],[]]
    sent_list_no=[[i,i+1] for i in range(len(sent_list)-2+1)]
    p=0
    jmp=0
    for i in sent_list_no:
        if p==1:
            p=0
            continue
        for j in [sent_list[i[0]],sent_list[i[1]]]:
                  if j.strip().split(" ")[-1] in tags:
                    p=0
                    jmp=1
                    break
        if jmp==1:
            jmp=0
            continue
        pat_score=ins.opt_inter_sent_pattern([pickle_list[i[0]],pickle_list[i[1]]],ref_pat,window_size,stop_words,tagid)
        avg_score=sum([k[1] for k in pat_score])/len(pat_score)
        if avg_score>=.80:
            p=1
            flag=1
            sent_list[i[0]]=sent_list[i[0]].strip()+" <"+label[1:-1].split("/")[0]+">\n"
            sent_list[i[1]]=sent_list[i[1]].strip()+" <"+label[1:-1].split("/")[1]+">\n"
            result_list[0].extend([sent_list[i[0],sent_list[i[1]]]])
        elif .70<avg_score<.80:
            result_list[1].append(pat_score)
        if flag==1:
            try:
                os.remove(file_name)
            except:
                pass
            with open(file_name,"a") as f:
                    f.write("".join(new_list))
            return {"tagged_sentences":result_list[0],"partial_matched_patterns":result_list[1]}
        else:
            return {"tagged_sentences":result_list[0],"partial_matched_patterns":result_list[1]}