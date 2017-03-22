#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

typedef float real;

struct vocab_word{
    long long cn;
    int *point;
    char *word;
    char *code;
    char codelen;
};

const int vocab_hash_size = 30000000;

int binary = 0; // 保存结果为binary
int cbow = 1; // 是否训练cbow 模型
int debug_mode = 2; // 是否是Debug 模式
int window = 5;     //
int min_count = 5;  // 最小词频
int num_threads = 12;  // 线程数
int min_reduce = 1;    //
int *vocab_hash;

long long vocab_max_size = 1000; // 总次数的初始值
long long vocab_size = 0;        // vocab 大小，词的总数
long long layer1_size = 100;    //
long long train_words = 0;
long long word_count_actual = 0;
long long iter = 5;             // 迭代次数
long long file_size = 0;       // 文件大小
long long classes = 0;         // 聚类数量

char train_file[MAX_STRING]; // 训练文件路径
char output_file[MAX_STRING]; //输出结果文件路径
char save_vocab_file[MAX_STRING]; // vocab 文件保存路径
char read_vocab_file[MAX_STRING]; // vocab 文件的读取路径
clock_t start;


real alpha = 0.025;
real starting_alpha;
real sample = 1e-3;

real *syn0;
real *syn1;
real *syn1neg;
real *expTable;


int hs = 0;
int negative = 5;
const int table_size = 1e8;
int *table;
struct vocab_word *vocab; // 保存vocab


/**
 * 解析命令行参数
 **/
int ArgPos(char *str, int argc, char **argv){
    int a ;
    for (a=1; a<argc; a++) if(!strcmp(str, argv[a])){
        if (a==argc-1){
            printf("Argument missing for %s\n", str);
        }
        return a;
    }
    return -1;
}

//word 的hash值
int GetWordHash(char *word){
    unsigned long long a, hash=0;
    for (a = 0; a<strlen(word); a++){
        hash = hash*257 + word[a];
    }
    hash = hash % vocab_hash_size;
    return hash;
}

// 添加word 到vocab 中
int AddWordToVocab(char *word){
    unsigned int hash;
    unsigned int length = strlen(word)+1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    if (vocab_size + 2 >= vocab_max_size){
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }

    hash = GetWordHash(word);
    // 如果hash 值重复
    while(vocab_hash[hash] !=-1) hash = (hash+1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size -1;
    return vocab_size-1;
}

void ReadWord(char *word, FILE *fin){
    int a=0;
    int ch ;
    while(!feof(fin)){
        ch = fgetc(fin);
        if (ch == 13) continue;
        if((ch==' ')|| (ch=='\t')||ch=='\n'){
            if(a>0){
                if(ch =='\n')ungetc(ch, fin);
                break;
            }
            if(ch=='\n'){
                strcpy(word, (char *)"</s>");
                return ;
            }else continue;
        }
        word[a] =ch;
        a++;
        if(a>=MAX_STRING-1) a--;
    }
    word[a] = 0;
}

int VocabCompare(const void *a, const void *b){
    return ((struct vocab_word *)b)->cn -((struct vocab_word *)a)->cn;
}


int SearchVocab(char *word){
    unsigned int hash = GetWordHash(word);
    while(1){
        if (vocab_hash[hash]==-1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash+1)%vocab_hash_size;
    }
    return -1;
}


void ReduceVocab(){
    int a,b = 0;
    unsigned int hash;
    for(a=0; a<vocab_size; a++) if (vocab[a].cn>min_reduce){
        vocab[b].cn = vocab[a].cn;
        vocab[b].word = vocab[a].word;
        b++;
    } else free(vocab[a].word);
    vocab_size=b;
    for(a = 0; a<vocab_hash_size; a++) vocab_hash[a] = -1;
    for(a=0; a<vocab_size; a++){
        hash = GetWordHash(vocab[a].word);
        while(vocab_hash[hash]!=-1) hash = (hash+1)%vocab_hash_size;
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

// 对vocab 排序
void SortVocab(){
    int a, size;
    unsigned int hash;
    qsort(&vocab[1], vocab_size-1, sizeof(struct vocab_word), VocabCompare);
    for(a=0; a<vocab_hash_size; a++) vocab_hash[a] = -1;
    size = vocab_size;
    train_words = 0;
    for(a =0; a<size; a++){
        if((vocab[a].cn<min_count)&&(a!=0)){
            vocab_size--;
            free(vocab[a].word);
        }else{
            hash = GetWordHash(vocab[a].word);
            while (vocab_hash[hash]!=-1) hash =(hash+1)%vocab_hash_size;
            vocab_hash[hash] = a;
            train_words += vocab[a].cn;
        }
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size+1)*sizeof(struct vocab_word));
    for(a=0; a<vocab_size; a++){
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

// 从训练数据中完成词频统计
void LearnVocabFromTrainFile(){
     char word[MAX_STRING];
     FILE *fin;
     long long a, i;
     //所有hash 赋值-1
     for (a = 0; a<vocab_hash_size; a++){
         vocab_hash[a] = -1;
     }
     fin = fopen(train_file, "rb");
     if (fin==NULL){
         printf("ERROR: training data file not found !\n");
         exit(1);
     }
     vocab_size = 0;
     AddWordToVocab((char *)"</s>");
     while(1){
         ReadWord(word, fin);
         if (feof(fin))break;
         train_words++;
         if((debug_mode>1)&& train_words%100000==0){
             printf("%lldk%c", train_words/1000, 13);
             fflush(stdout);
         }
         i = SearchVocab(word);
         if (i==-1){
             a = AddWordToVocab(word);
             vocab[a].cn = 1;
         }else vocab[i].cn++;
         if (vocab_size>vocab_hash_size*0.7) ReduceVocab();
     }
     SortVocab();
     if(debug_mode>0){
         printf("Vocab size:%lld\n", vocab_size);
         printf("words in train file:%lld\n", train_words);
     }
     file_size = ftell(fin);
     fclose(fin);
}


void SaveVocab(){
    long long i;
    FILE *fo = fopen(save_vocab_file , "wb");
    for (i=0; i<vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void CreateBinaryTree(){

}

void InitNet(){
    long long a, b;
    unsigned long long next_random=1;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size *layer1_size*sizeof(real));
    if(syn0==NULL){
        printf("Memory allocation failed\n");
        exit(1);
    }
    if(hs){
        a = posix_memalign((void **)&syn1, 128, (long long)vocab_size*layer1_size *sizeof(real));
        if(syn1==NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for( a= 0; a<vocab_size; a++) for(b=0; b<layer1_size; b++){
            next_random = next_random *(unsigned long long)25214903917 +11;
            syn0[a*layer1_size+b] = (((next_random & 0xFFFF)/ (real)65535)-0.5)/layer1_size;
        }
    }
    CreateBinaryTree();

}

void TrainModel(){
    long a, b, c, d;
    FILE *fo;
    printf("String training using file %s\n", train_file);
    starting_alpha = alpha;

    if(read_vocab_file[0]!=0){
        // 直接读取计算好的vocab
        //ReadVocab();
    }
    else{
        // 从训练语料中产生vocab
        LearnVocabFromTrainFile();
    }
    if(save_vocab_file[0]!=0)SaveVocab();
    if(output_file[0]==0)return;
    // 初始化网络
    InitNet();
}


int main(int argc, char **argv){
    int i;

    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
    if (cbow) alpha = 0.05;
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    expTable = (real *)calloc((EXP_TABLE_SIZE-1), sizeof(real));
    for (i=0; i<EXP_TABLE_SIZE; i++){
        expTable[i] = exp((i/(real)EXP_TABLE_SIZE*2-1)*MAX_EXP);
        expTable[i]= expTable[i] / (expTable[i]+1);
    }
    TrainModel();
    return 0;
}

