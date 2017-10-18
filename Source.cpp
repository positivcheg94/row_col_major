#include <mkl.h>

int main()
{
    // 2 * 4 || 4 * 2
    int m = 2, n = 3, k = 4;
    double A[] = { 1,2,3,4,5,6,7,8 };
    // 4 * 3 || 3 * 4
    double B[] = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    // 2 * 3
    double C11[] = { 0,0,0,0,0,0 };
    double C12[] = { 0,0,0,0,0,0 };
    double C13[] = { 0,0,0,0,0,0 };
    double C14[] = { 0,0,0,0,0,0 };

    // A , B
    // col major 2 * 4 matrix @ col major B 4 * 3
    cblas_dgemm(CblasColMajor, CblasNoTrans , CblasNoTrans  , m, n, k, 1, A, m, B, k, 0, C11, m);
    // row major 2 * 4 matrix @ col major B 4 * 3
    cblas_dgemm(CblasColMajor, CblasTrans   , CblasNoTrans  , m, n, k, 1, A, k, B, k, 0, C12, m);

    // col major 2 * 4 matrix @ row major B 4 * 3
    cblas_dgemm(CblasColMajor, CblasNoTrans , CblasTrans    , m, n, k, 1, A, m, B, n, 0, C13, m);

    // row major 2 * 4 matrix @ row major B 4 * 3
    cblas_dgemm(CblasColMajor, CblasTrans   , CblasTrans    , m, n, k, 1, A, k, B, n, 0, C14, m);




    double C21[] = { 0,0,0,0,0,0 };
    double C22[] = { 0,0,0,0,0,0 };
    double C23[] = { 0,0,0,0,0,0 };
    double C24[] = { 0,0,0,0,0,0 };
    // A            , B transpose
    // col major 2 * 4 matrix @ col major B 3 * 4 T
    cblas_dgemm(CblasColMajor, CblasNoTrans , CblasTrans    , m, n, k, 1, A, m, B, n, 0, C21, m);
    // row major 2 * 4 matrix @ col major B 3 * 4 T
    cblas_dgemm(CblasColMajor, CblasTrans   , CblasTrans    , m, n, k, 1, A, k, B, n, 0, C22, m);

    // col major 2 * 4 matrix @ row major B 3 * 4 T
    cblas_dgemm(CblasColMajor, CblasNoTrans , CblasNoTrans  , m, n, k, 1, A, m, B, k, 0, C23, m);

    // row major 2 * 4 matrix @ row major B 3 * 4 T
    cblas_dgemm(CblasColMajor, CblasTrans   , CblasNoTrans  , m, n, k, 1, A, k, B, k, 0, C24, m);

    double C31[] = { 0,0,0,0,0,0 };
    double C32[] = { 0,0,0,0,0,0 };
    double C33[] = { 0,0,0,0,0,0 };
    double C34[] = { 0,0,0,0,0,0 };
    // A transpose  , B
    // col major 4 * 2 matrix T @ col major B 4 * 3
    cblas_dgemm(CblasColMajor, CblasTrans   , CblasNoTrans  , m, n, k, 1, A, k, B, k, 0, C31, m);
    // row major 4 * 2 matrix T @ col major B 4 * 3
    cblas_dgemm(CblasColMajor, CblasNoTrans , CblasNoTrans  , m, n, k, 1, A, m, B, k, 0, C32, m);

    // col major 4 * 2 matrix T @ row major B 4 * 3
    cblas_dgemm(CblasColMajor, CblasTrans   , CblasTrans    , m, n, k, 1, A, k, B, n, 0, C33, m);

    // row major 4 * 2 matrix T @ row major B 4 * 3
    cblas_dgemm(CblasColMajor, CblasNoTrans , CblasTrans    , m, n, k, 1, A, m, B, n, 0, C34, m);

    return 0;
}