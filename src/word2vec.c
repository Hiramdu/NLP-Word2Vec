//
//  main.c
//  word2vec
//
//  Created by apple on 2017/7/13.
//  Copyright © 2017年 apple. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>  
//#include <omp.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;

typedef float real;

struct vocab_word
{
    long long cn;
    int *point;
    char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;
int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

void initunigramtable()
{
    int a,i;
    long long train_words_pow=0;
    real power=0.75;
    real dl;
    for(a=0;a<vocab_size;a++)
    train_words_pow+=pow(vocab[a].cn,power);
    i=0;
    dl=pow(vocab[i].cn,power)/(real)train_words_pow;
    for(a=0;a<table_size;a++)
    {
        table[a]=i;
        if(a/(real)table_size>dl)
        {
            i++;
            dl+=pow(vocab[i].cn,power)/(real)train_words_pow;
        }
        if(i>=vocab_size)
            i=vocab_size-1;
    }
}

void readword(FILE *file,char *word){
    int flag=0;
    int ch;
    if(feof(&file)==0)
    { ch=fgetc(&file);
        if(ch==13)
        //    continue;
        if((ch=='\n')||(ch==' ')||(ch=='\t'))
        {
           if(flag>0)
               if(ch=='\n')
                   ungetc(ch,&file);
            //   break;
        }
        if(ch=='\n'){
            strcpy(word,(char *)"</s>");
            return;
        }
        //else  continue;
    }
    word[flag]=ch;
    flag++;
    if(flag>=MAX_STRING-1)
        flag--;
    word[flag]=0;
}

int getwordhash(char *word){
    unsigned long i;
    long hash=0;
    for(i=0;i<strlen(*word);i++)
    hash=hash*257+word[i];
    return hash;
}

int searchvocab(char *word){
    unsigned long long hash=getwordhash(word);
  //  long long vocab_hash;
    while(1)
    {
        if(vocab_hash[hash]==-1)
            return -1;
        if(!strcpy(word,vocab[vocab_hash[hash]].word))
            return vocab_hash[hash];
            hash=(hash+1)%vocab_hash_size;
    }
    return -1;
}

int readwordindex(FILE file){
    char word[MAX_STRING];
    readword(&file,*word);
    if(feof(&file)!=0) return -1;
    return searchvocab(word);
}

int addwordtovocab(char word){
    unsigned int hash,length=strlen(word);
    if(length>MAX_STRING)
        length=MAX_STRING;
    vocab[vocab_size].word=(char *)calloc(length,sizeof(char));
    strcpy(vocab[vocab_size].word,word);
    vocab[vocab_size].cn=0;
    vocab_size++;
    
    hash=getwordhash(word);
    while(vocab_hash[hash]!=-1)
        hash=(hash+1) % vocab_hash_size;
        vocab_hash[hash]=vocab_size-1;
    return vocab_hash[hash];
}

void sortvocab(){
    int a,size;
    unsigned int hash;
   // qsort(&vocab[1],vocab_size-1,sizeof(struct vocab_word),VocabCompare);
    for(a=0;a<vocab_hash_size;a++)
        vocab_hash[a]=-1;
    size=vocab_size;
    train_words=0;
    for(a=0;a<size;a++)
    {
        if(vocab[a].cn<min_count)
        {
            vocab_size--;
            free(vocab[a].word);
        }
        else
        {
            hash=getwordhash(vocab[a].word);
            while(vocab_hash[hash]!=-1)
                hash=(hash+1)%vocab_hash_size;
            vocab_hash[hash]=a;
            train_words+=vocab[a].cn;
        }
    }
    vocab=(struct vocab_word *)realloc(vocab,(vocab_size+1)*sizeof(struct vocab_word));
    for(a=0;a<vocab_size;a++)
    {
        vocab[a].code=(char *)calloc(MAX_CODE_LENGTH,sizeof(char));
        vocab[a].point=(int *)calloc(MAX_CODE_LENGTH,sizeof(int));
    }
}

void reducevocab(){
    int a,b=0;
    unsigned int hash;
    for(a=0;a<vocab_size;a++)
        if(vocab[a].cn>min_reduce)
        {
            vocab[b].cn=vocab[a].cn;
            vocab[b].word=vocab[a].word;
            b++;
        }
        else free(vocab[a].word);
    vocab_size=b;
    for(a=0;a<vocab_hash_size;a++)
        vocab_hash[a]=-1;
    for(a=0;a<vocab_size;a++)
    {
        hash=getwordhash(vocab[a].word);
        while(vocab_hash[hash]!=-1)
            hash=(hash+1)%vocab_hash_size;
        vocab_hash[hash]=a;
    }
    min_reduce++;
}

void createtree(){
    long long a,b,i,pos1,pos2,min1,min2,point[MAX_CODE_LENGTH];
    long long *count=(long long *)calloc(vocab_size*2+1,sizeof(long long));
    char code[MAX_CODE_LENGTH];
    long long *parent_node=(long long *)calloc(vocab_size*2+1,sizeof(long long));
    long long *binary=(long long *)calloc(vocab_size*2+1,sizeof(long long));
    for(a=0;a<vocab_size-1;a++){
        if(pos1>=0) {
            if (count[pos1] > count[pos2]) {
                min1 = pos2;
                pos2++;
            } else {
                min1 = pos1;
                pos1--;
            }
        } else{
                min1=pos2;
                pos2++;
            }
          if(pos1>=0)
          {
              if(count[pos1]<count[pos2])
              {   min2=pos1;
                  pos1--;
              }
              else{
                  min2=pos2;
                  pos2++;
              }
          }
            else{
                min2=pos2;
                pos2++;
            }
            count[vocab_size+a]=count[min1]+count[min2];
            parent_node[min1]=vocab_size+a;
            parent_node[min2]=vocab_size+a;
            binary[min2]=1;
}
    for(a=0;a<vocab_size;a++)
    {
        b=a;
        i=0;
        while(1)
        {
            code[i]=binary[b];
            point[i]=b;
            i++;
            b=parent_node[b];
            if(b==vocab_size*2-2) break;
        }
        vocab[a].codelen=i;
        vocab[a].point[0]=vocab_size-2;
        for(b=0;b<i;b++)
        {
            vocab[a].code[i-b-1]=code[b];
            vocab[a].point[i-b]=point[b]-vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

void learnvocabfromfile()
{
    int a;
    long long i;
    FILE *file;
    char word[MAX_STRING];
    for(a=0;a<vocab_size;a++)
        vocab_hash[a]=-1;
    file=fopen(train_file,"rb");
    while(1)
    {
        readword(file,*word);
        if(feof(&file)) break;
        i=searchvocab(word);
        if(i==-1)
        {            a=addwordtovocab(word);
            vocab[a].cn=1; }
        else vocab[i].cn++;
        if(vocab_size>vocab_hash_size*0.7)
            reducevocab();
        sortvocab();
        fclose(file);
    }
}

void savevocab()
{
    int i;
    FILE *file=fopen(save_vocab_file,"wb");
    for(i=0;i<vocab_size;i++)
        fprintf(file,vocab[i].word,vocab[i].cn);
    fclose(file);
}

void readvocab()
{
    FILE *file=fopen(read_vocab_file,'rb');
    int a,i=0;
    char word[MAX_STRING];
    char c;
    if(file==NULL)
    {
        printf("voabulary file not found");
        exit(1);
    }
    for(a=0;a<vocab_hash_size;a++)
        vocab_hash[a]=-1;
    vocab_size=0;
    while(1)
    {
        readword(file,*word);
        if(feof(file)) break;
           a=addwordtovocab(word);
           fscanf(file,vocab[a].cn,&c);
           i++;
    }
           sortvocab();
           file=fopen(train_file,"rb");
           if(file==NULL)
           {
               printf("error:training file not found");
               exit(1);
           }
           fseek(file,0,SEEK_END);
           file_size=ftell(file);
           fclose(file);
}

void InitNet()
        {
    int a,b;
    if(syn0==NULL)
    {
        printf("memory location fail");
        exit(1);
    }
    if(hs)
    {
        a=posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if(syn1==NULL)
        {
            printf("memory location fail");
            exit(1);
        }
        for(b=0;b<layer1_size;b++)
            for(a=0;a<vocab_size;a++)
                syn1neg[a*layer1_size+b]=0;
    }
            for(b=0;b<layer1_size;b++)
                for(a=0;a<vocab_size;a++)
                    syn0[a*layer1_size+b]=(rand()/(real)RAND_MAX-0.5)/layer1_size;
            createtree();
        }

void trainmodelthread(void *id)
           {
               long long a,b,d,word,last_word,sentence_length=0,sentence_position=0;
               long long word_count=0,last_word_count=0,sen[MAX_SENTENCE_LENGTH+1];
               long long l1,l2,c,target,label;
               unsigned long long next_random=(long long)id;
               real f,g;
               clock_t now;
               real *neu1=(real *)calloc(layer1_size,sizeof(real));
               real * neu1e=(real *)calloc(layer1_size,sizeof(real));
               FILE *file=fopen(train_file,"rb");
               fseek(file,file_size/(long long)num_threads*(long long)id,SEEK_SET);
               while(1)
               {
                   if(word_count-last_word_count>10000)
                   {
                       word_count_actual+=word_count-last_word_count;
                       last_word_count=word_count;
                       if(debug_mode>1)
                       {
                           now=clock();
                           printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                                  word_count_actual / (real)(train_words + 1) * 100,
                                  word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                           fflush(stdout);
                       }
                       alpha=starting_alpha*(1-word_count_actual/(real)(train_words+1));
                       if(alpha<starting_alpha*0.0001)
                           alpha=starting_alpha*0.0001;
                   }
                   if(sentence_length==0)
                   {
                       while(1)
                       {
                           word=readwordindex(*file);
                           if(feof(file)) break;
                           if(word==0) break;
                           if(word==-1) continue;
                           word_count++;
                           if(sample>0)
                           {
                               real ran=(sqrt(vocab[word].cn/(sample * train_words))+1)*(sample*train_words)/vocab[word].cn;
                               next_random=next_random*(unsigned long long)25214903917+l1;
                               if(ran<(next_random & 0xFFFF)/(real)65536) continue;
                           }
                           sen[sentence_length]=word;
                           sentence_length++;
                           if(sentence_length>=MAX_SENTENCE_LENGTH) break;
                       }
                       sentence_position=0;
                   }
                   if(feof(file)) break;
                   if(word_count>train_words/num_threads) break;
                   word=sen[sentence_position];
                   if(word==-1) continue;
                   for(c=0;c<layer1_size;c++) neu1[c]=0;
                   for(c=0;c<layer1_size;c++) neu1e[c]=0;
                   next_random=next_random*(unsigned long long)25214903917+l1;
                   b=next_random % window;
                   if(cbow)
                   {
                       for(a=b;a<window*2+1-b;a++)
                           if(a!=window)
                           {
                               c=sentence_position-window+a;
                               if(c>=sentence_length) continue;
                               last_word=sen[c];
                               if(c<0) continue;
                               if(last_word==-1) continue;
                               for(c=0;c<layer1_size;c++)
                                 neu1[c]+=syn0[c+last_word*layer1_size];
                           }
                       if(hs)
                           for(d=0;d<vocab[word].codelen;d++)
                           {
                               f=0;
                               l2=vocab[word].point[d]*layer1_size;
                               for(c=0;c<layer1_size;c++)
                                   f+=neu1[c]*syn1[c+l2];
                               if(f<=-MAX_EXP) continue;
                               else if(f>=MAX_EXP) continue;
                               else f=expTable[(int)((f+MAX_EXP)*(EXP_TABLE_SIZE/MAX_EXP/2))];
                               g=(1-vocab[word].code[d]-f)*alpha;
                               for(c=0;c<layer1_size;c++)
                                   neu1e[c]+=g*syn1[c+l2];
                               for(c=0;c<layer1_size;c++)
                                   syn1[c+l2]+=g*neu1[c];
                           }
                       if(negative>0)
                           for(d=0;d<negative+1;d++)
                           {
                               if(d==0)
                               {
                                   target=word;
                                   label=1;
                               }
                               else
                               {
                                   next_random=next_random*(unsigned long long)25214903917+l1;
                                   target=table[(next_random>>16)%table_size];
                                   if(target==0)
                                       target=next_random%(vocab_size-1)+1;
                                   if(target==word) continue;
                                   label=0;
                               }
                               l2=target*layer1_size;
                               f=0;
                               for(c=0;c<layer1_size;c++)
                                   f+=syn0[c+l1]*syn1neg[c+l2];
                               if(f>MAX_EXP)
                                   g=(label-1)*alpha;
                               else if(f<-MAX_EXP)
                                   g=label*alpha;
                               else g=label-expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))] * alpha;
                            for(c=0;c<layer1_size;c++)
                                neu1e[c]+=g*syn1neg[c+l2];
                               for(c=0;c<layer1_size;c++)
                                   syn1neg[c+l2]+=g*syn0[c+l1];
                           }
                       for(c=0;c<layer1_size;c++)
                           syn0[c+l1]+=neu1e[c];
                   }
               }
               sentence_position++;
               if(sentence_position>=sentence_length)
               {
                   sentence_length=0;
                 //  continue;
               }
               fclose(file);
               free(neu1);
               free(neu1e);
               pthread_exit(NULL);
           }

void trainmodel()
{
    FILE file;
    printf("start train using file",train_file);
    long a,b,c,d;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    if(read_vocab_file[0]!=0)
        readvocab();
    else
        learnvocabfromfile();
    if(save_vocab_file[0]!=0)
        savevoab();
    if(output_file[0]==0)
        return;
    initnet();
    if(negative>0)
        initunigramtable();
    start=clock();
    for(a=0;a<num_threads;a++)
        pthread_create(&pt[a], NULL, trainmodelthread, (void *)a);
    for(a=0;a<num_threads;a++)
            pthread_join(pt[a],NULL);
    file=*fopen(output_file,"wb");
    if(classes==0)
    {
        fprintf(&file,"%lld %lld\n",vocab_size,layer1_size);
        for(a=0;a<vocab_size;a++)
        {
        fprintf(&file,"%s",vocab[a].word);
            if(binary)
                for(b=0;b<layer1_size;b++)
                    fwrite(&syn0[a*layer1_size+b],sizeof(real),1,&file);
            else
                for(b=0;b<layer1_size;b++)
                    fprintf(&file,"%lf",syn0[a*layer1_size+b]);
            fprintf(&file,"\n");
        }
    }
    else
    {
    int clcn=classes,iter=10,closeid;
        int *centcn=(int *)malloc(classes*sizeof(int));
        int *cl=(int)calloc(vocab_size,sizeof(int));
        real closev,x;
        real *cent=(real *)calloc(classes*layer1_size,sizeof(real));
        for(a=0;a<vocab_size;a++)
            cl[a]=a%clcn;
        for(a=0;a<iter;a++)
        {
            for(b=0;b<clcn*layer1_size;b++)
                cent[b]=0;
            for(b=0;b<clcn;b++)
                centcn[b]=1;
            for(c=0;c<vocab_size;c++)
            {
                for(d=0;d<layer1_size;d++)
                    cent[layer1_size*cl[c]+d]+=syn0[c*layer1_size+d];
                centcn[cl[c]]++;
            }
            for(b=0;b<clcn;b++)
            {
                closev=0;
                for(c=0;c<layer1_size;c++)
                {
                    cent[layer1_size*b+c]/=centcn[b];
                    closev+=cent[layer1_size*b+c]*cent[layer1_size*b+c];
                }
                closev=sqrt(closev);
                for(c=0;c<layer1_size;c++)
                    cent[layer1_size*b+c]/=closev;
            }
            for(c=0;c<vocab_size;c++)
            {
                closev=-10;
                closeid=0;
                for(d=0;d<clcn;d++)
                {
                    x=0;
                    for(b=0;b<layer1_size;b++)
                        x+=cent[layer1_size*d+b]*syn0[c*layer1_size+b];
                    if(x>closev)
                    {
                        closev=x;
                        closeid=d;
                    }
                }
                cl[c]=closeid;
            }
        }
        for(a=0;a<vocab_size;a++)
            fprintf(&file, "%s %d\n", vocab[a].word, cl[a]);
        free(centcn);
        free(cent);
        free(cl);
    }
    fclose(&file);
}

int argpos(char *str,int argc,char **argv)
{
    int a;
    for(a=1;a<argc;a++)
        if(!strcmp(str,argv[a]))
        {
            if(a==argc-1)
            {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");

        //输入文件：已分词的语料
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");

        //输出文件：词向量或者词聚类
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");

        //词向量的维度，默认值是100
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");

        //窗口大小，默认是5
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");

        //设定词出现频率的阈值，对于常出现的词会被随机下采样
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
        printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");

        //是否采用softmax体系
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");

        //负样本的数量，默认是0，通常使用5-10。0表示不使用。
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");

        //开启的线程数量
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");

        //最小阈值。对于出现次数少于该值的词，会被抛弃掉。
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");

        //学习速率初始值，默认是0.025
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");

        //输出词类别，而不是词向量
        printf("\t-classes <int>\n");
        printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");

        //debug模式，默认是2，表示在训练过程中会输出更多信息
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");

        //是否用binary模式保存数据，默认是0，表示否。
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");

        //保存词汇到这个文件
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");

        //词汇从该文件读取，而不是由训练数据重组
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");

        //是否采用continuous bag of words算法。默认是0，表示采用另一个叫skip-gram的算法。
        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");

        //工具使用样例
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
        return 0;
    }
    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++)
    {
        //expTable[i] = exp((i -500)/ 500 * 6) 即 e^-6 ~ e^6
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        //expTable[i] = 1/(1+e^6) ~ 1/(1+e^-6)即 0.01 ~ 1 的样子
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    TrainModel();
    return 0;
}



















