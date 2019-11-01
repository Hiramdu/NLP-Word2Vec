#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

const long long max_size=2000;
const long long N=40;
const long long max_w=50;

int main(int argc,char **argv) {
    long long words, size;
    char vocab, ch;
    long long a, b;
    float M;
    if (argc <= 1) {
        printf("error");
        return 0;
    }
    strcpy(file_name, argv[1]);
    FILE file = fopen(file_name, argv[1]);
    fscanf(file, words);
    fscanf(file, size);
    for (b = 0; b < words; b++) {
        fscanf(file, vocab[b + max_w], ch);
        for (int a = 0; a < size; a++)
            fread(M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++)
        len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++)
            M[a + b * size] /= len;
    }
    flose(file);
    while (1) {
        for (a = 0; a < N; a++)
            bestd[a] = 0;
        for (a = 0; a < N; a++)
            bestw[a][0] = 0;
        printf("Enter word:");
        a = 0;
        while (1) {
            st1[a] = fgetc(stdin);
            if ((st1[a] == '\n') || (a >= max_size - 1)) {
                st1[a] = 0;
                break;
            }
            a++;
        }
        if (!strcmp(st1, "EXIT"))
            break;
        cn = 0;
        b = 0;
        c = 0;
        while (1) {
            st[cn][b] = st1[c];
            b++;
            c++;
            st[cn][b] = 0;
            if (st1[c] == 0)
                break;
            if (st1[c] == ' ') {
                cn++;
                b = 0;
                c++;
            }
        }
        cn++;
        for (a = 0; a < cn; a++) {
            for (b = 0; b < words; b++)
                if (!strcmp(vocab[b * max_w], st[a]))
                    break;
            if (b == words)
                b = -1;
            bi[a] = b;
            printf("position in vocab", st[a], bi[a]);

        }
        if (b == -1)
            continue;
        printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
        for (a = 0; a < size; a++)
            vec[a] = 0;
        for (b = 0; b < cn; b++) {
            if (bi[b] == -1)
                continue;
            for (a = 0; a < size; a++)
                vec[a] += M[a + bi[b] * size];
        }
        len = 0;
        for (a = 0; a < size; a++)
            len += vec[a] * vec[a];
        len = sqrt(len);
        for (a = 0; a < size; a++)
            vec[a] /= len;
        for (a = 0; a < N; a++)
            bestd[a] = 0;
        for (a = 0; a < N; a++)
            bestw[a][0] = 0;
        for (c = 0; c < words; c++) {
            a = 0;
            for (b = 0; b < cn; b++)
                if (bi[b] == c)
                    a = 1;
            if (a == 1)
                continue;
            dist = 0;
            for (a = 0; a < size; a++)
                dist += vec[a] * M[a + c * size];
            for (a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (d = N - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], vocab[c * max_w]);
                    break;
                }
            }
        }
        for (a = 0; a < N; a++)
            printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    }
    return 0;

}





















