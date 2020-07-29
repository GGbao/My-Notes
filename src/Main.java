import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Main {
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int k = sc.nextInt();
//        int[] arr = new int[n];
//        for (int i = 0; i < n; i++) {
//            arr[i] = sc.nextInt();
//        }
//
//        if (n % 2 == 0) {
//            for (int i = 0; i < k; i++) {
//                arr = replace(arr, n);
//                for(int j=0;j<arr.length;j++) {
//                    System.out.print(arr[j] + " ");//输出数组
//                }
//            }
//        }
//
//
//    }

    /*public static Lock lock = new ReentrantLock();
    public static int state = 0;

    static class myThread1 extends Thread {
        public void run() {
            for (int i = 0; i < 10; ) {
                try {
                    lock.lock();
                    while (state % 3 == 0) {
                        System.out.println("a");
                        i++;
                        state++;
                    }
                } finally {
                    lock.unlock();
                }
            }
        }

    }

    static class myThread2 extends Thread {
        public void run() {
            for (int i = state; i < 10; ) {
                try {
                    lock.lock();
                    while (state % 3 == 1) {
                        System.out.println("l");
                        i++;
                        state++;
                    }
                } finally {
                    lock.unlock();
                }
            }
        }

    }

    static class myThread3 extends Thread {
        public void run() {
            for (int i = state; i < 10; ) {
                try {
                    lock.lock();
                    while (state % 3 == 2) {
                        System.out.println("i");
                        i++;
                        state++;
                    }
                } finally {
                    lock.unlock();
                }
            }
        }

    }*/

    public static void main(String[] args) {

        /*new myThread1().start();
        new myThread2().start();
        new myThread3().start();*/
        //01背包
        /*
        有 N 件物品和一个容量是 V 的背包。每件物品只能使用一次。
        第 i 件物品的体积是 vi，价值是 wi。
        求解将哪些物品装入背包，可使这些物品的总体积不超过背包容量，且总价值最大。
        输出最大价值。
        * */
        /*f[i][j]表示只看前i个物品，总体积是j的情况下，总价值最大多少
        * result=max{f[n][0~V]}
        * f[i][j]:
        * 1.不选第i个物品，f[i][j]=f[i-1][j];
        * 2.选第i个物品，f[i][j]=f[i-1][j-v[i]]+w[i]
        * f[i][j]=max{1. 2.}
        * f[0][0]=0;
        * */

        /**
         * 包粽子，包一个纯面粉的粽子需要c 克面粉，可以卖出 d 块钱
         * 有m种配料，每种配料可以对应包一种粽子，比如
         * 第i种配料有a[i]克，包一个该配料的粽子需要配料b[i]克，面粉c[i]克，可以卖出d[i]块钱
         * 问，有n克面粉，m种配料，最多可以包粽子卖出多少块钱？
         * 输入第一行为
         * n m c d
         * 表示n克面粉，m种配料，纯面粉粽子需要c克面粉，价值为d
         * 接下来m行，每行四个数a[i] b[i] c[i] d[i]  ，分别代表该配料的总重量，包一个粽子需要的配料和面粉以及价值
         * 把纯面的粽子当成一种配料处理
         * n 0 c d
         **/
        //dp[i][j]   表示使用前 i 种配料，消耗 j 克面粉的情况下的最大价值为dp[i][j]
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int m = sc.nextInt();
        int[] a = new int[m+1];
        int[] b = new int[m+1];
        int[] c = new int[m+1];
        int[] d = new int[m+1];
        //把纯面的粽子当成一种配料处理
        a[0] = m;
        b[0] = 0;
        c[0] = sc.nextInt();
        d[0] = sc.nextInt();
        int[][] f=new int[m+1][n+1];

        for (int i = 1; i <= m; i++) {
            a[i] = sc.nextInt();
            b[i] = sc.nextInt();
            c[i] = sc.nextInt();
            d[i] = sc.nextInt();
        }
        //第0种粽子
        for (int i = 0; i <= n; i++) {
            f[0][i]=i/c[0]*d[0];
        }
        //前i种粽子
        for (int i = 1; i <= m; i++) {
            //消耗j克面粉
            for (int j = 0; j <= n; j++) {
                f[i][j] = f[i - 1][j];
                //在面粉够用的情况下，最多包k个第i种配料的粽子
                for (int k = 1; k*c[i]<=j ; k++) {
                    //还要保证配料够
                    if (a[i]>=b[i]*k) {
                        f[i][j] = Math.max(f[i][j],f[i - 1][j - c[i]*k] + d[i]*k);
                    }
                }
            }
        }
        System.out.println(f[m][n]);
        }
    }







