import java.io.BufferedInputStream;
import java.util.*;

public class alibaba {
    //给定一个n，求[1,n]这n个数字的排列组合有多少个。条件：相邻的两个数字的绝对值不能等于1
    //思路：简单的回溯算法，注意保存上一次访问的位置用于判定绝对值\
    /*
    Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] num = new int[n];
        for (int i = 0; i < n; i++) {
            num[i] = i+1;
        }
        List<List<Integer>> res = new ArrayList<>();
        judge(n, res,num, new LinkedList<>());
        for (List<Integer> list : res) {
            System.out.println(list);

        }
        */
    private static void judge(int len, List<List<Integer>> res, int[] nums, LinkedList<Integer> stack) {
        if (stack.size() == len && findArr(stack)) {

            res.add(new ArrayList<>(stack));
            return;
        }
        for (int j = 0; j < nums.length; j++) {
            if (stack.contains(nums[j])) {
                continue;
            }
            stack.add(nums[j]);
            judge(len, res, nums, stack);
            stack.removeLast();

        }

    }

    private static boolean findArr(LinkedList<Integer> stack) {
        int n = stack.size();
        for (int i = 1; i < n; i++) {
            if (Math.abs(stack.get(i) - stack.get(i - 1)) <= 1) {
                return false;
            }
        }
        return true;
    }

    //题目1:给定一个数x，数据对 (a, b)使得a ^ b ^ x能达到最大，求使|a - b|最小的方案总数有多少。
    //x,a,b的范围都是0 - （2^31 次方-1）
    //x^a^b=INT_MAX,就是除去符号位每一位都是1,遍历x的每个bit，
    // 如果是1，那么a,b在此位的bit必相等，因为1^1^1=1,1^0^0=1，
    // 而相等的话对它们的差就没有贡献了，所以0、1皆可，方案数*2。
    // 对于bit为0的情况，差值取最小的话是确定的。最后由于绝对值的对称性*2。
    // 特殊情况x=INTMAX，此时x的每一位已经全是1了，所以a-b取最小其实就是所有a=b，方案数INT-MAX + 1,用long存储输出。
    public long findSum(int x) {
        long res = 2;
        for (int i = 0; i < 32; i++) {
            if (((x << i) & 1) != 0) {
                res *= 2;
            }
        }
        if (x == Integer.MAX_VALUE) {
            return res / 2;
        } else {
            return res;
        }
    }

    //吃饼问题

    public void Cake(String[] args) {
        Scanner sc = new Scanner(System.in);
        int num = sc.nextInt();
        long[] arrs = new long[num];
        long res = 0;
        long min = Long.MAX_VALUE;
        for (int i = 0; i < num; i++) {
            arrs[i] = sc.nextLong();
            min = Math.min(arrs[i], min);
            res += min;
        }
        sc.close();
        System.out.println(res);

    }

    //开关灯。N行L列的灯，有L个开关，第i个开关Li可以控制第i列，打开该开关使得该列灯状态反转。
    //行之间可以任意交换，问给定初始灯状态s和目标灯状态t，能否从初始变到目标状态，如果能，最少要打开几个开关。
    /**
        第一行有一个整数T，表示有多少组测试数据。
        每组测试数据包含三行。第一行为两个整数n , L。
        每组数据的第二行为n个长度为LLL的0/1字符串，依次描述起初每行的灯的开关状态。
        第i个字符串的第jjj个字符若是’1’，表示对应位置的灯是亮的；’0’表示是灭的。
        每组数据的第三行为n个长度为L的0/1字符串，描述主办方希望达到的所有灯的开关状态。格式同上。

    **/
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int T = in.nextInt();
        for (int t = 0; t < T; t++) {
            lights(in);
        }
        in.close();

    }


    //输入矩阵方式
    public static void lights(Scanner in) {
        /**
         * 先读取输入
         * 行数多一行空 着等 后面放入这一行有多少个1
         * 输入矩阵方式
         */
        int N = in.nextInt();
        int L = in.nextInt();
        int[][] lights = new int[N + 1][L];
        for (int i = 0; i < N; i++) {
            String line = in.next();
            for (int j = 0; j < L; j++) {
                int temp = line.charAt(j) - '0';
                lights[i][j] = temp;
            }
        }
        int[][] target = new int[N + 1][L];
        for (int i = 0; i < N; i++) {
            String line = in.next();
            for (int j = 0; j < L; j++) {
                int temp = line.charAt(j) - '0';
                target[i][j] = temp;
            }
        }
        /**
         * END 输入读取
         */

        //添加纵列信息（在每一列最后一行添加这一列一共有多少个1）
        for (int j = 0; j < L; j++) {
            int trueNum = 0, tarTrueNum = 0;
            for (int i = 0; i < N; i++) {
                if (lights[i][j] == 1) {
                    trueNum++;
                }
                if (target[i][j] == 1) {
                    tarTrueNum++;
                }
            }
            lights[N][j] = trueNum;
            target[N][j] = tarTrueNum;
        }
        //开关次数
        int times = 0;
        /**
         * 通过比较每一列最后一行的信息（即此列有多少个一来判断是否可以通过开关灯实现）
         * 有相同个数1或者 1与目标0个数相同（按开关） 即可行，均不满足则不可能
         * 在比较过程中把需要按开关的列都按下开关
         */
        for (int i = 0; i < L; i++) {
            //有列不满足则直接返回
            if (lights[N][i] != target[N][i] && lights[N][i] != (N - target[N][i])) {
                System.out.println("Impossible");
                return;
            }
            //有需要按开关的则按开关，并对开关计次
            if (lights[N][i] != target[N][i] && lights[N][i] == (N - target[N][i])) {
                //对开关计次
                times++;
                //对值异或1使其反转
                for (int j = 0; j < N; j++) {
                    lights[j][i] ^= 1;
                }
            }
        }
        //再看横向，把横向一个数组转化成一个字符串 如”010010“
        String[] sLights = new String[N];
        String[] sTarget = new String[N];
        for (int i = 0; i < N; i++) {
            StringBuffer b1 = new StringBuffer();
            StringBuffer b2 = new StringBuffer();
            for (int j = 0; j < L; j++) {
                b1.append(String.valueOf(lights[i][j]));
                b2.append(String.valueOf(target[i][j]));
            }
            sLights[i] = b1.toString();
            sTarget[i] = b2.toString();
        }

        //再把字符串看成二进制转化为long（因为最多有50个数，超过了int的范围，只能用long）
        long[] lLights = new long[N];
        long[] lTarget = new long[N];
        for (int i = 0; i < N; i++) {
            lLights[i] = Long.valueOf(sLights[i], 2);
            lTarget[i] = Long.valueOf(sTarget[i], 2);
        }

        //对两个long的数组排序
        Arrays.sort(lLights);
        Arrays.sort(lTarget);

        //挨个比较long的值，如果有不等则返回，全相等则输出按开关的次数
        for (int i = 0; i < N; i++) {
            if (lLights[i] != lTarget[i]) {
                System.out.println("Impossible");
                return;
            }
        }
        System.out.println(times);
        return;

    }
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
    public static void zongzi(String[] args) {

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



    //n 由三个互不相等的数相而得，这三个数两两的最大公约数是k，1 <= k <= n <= 10^18。
    //输入：T组数据，每行给定n和k。
    //输出：是否存在这样三个数，存在则输出任意一组答案（n = x + y +z）,不存在则输出-1。

    private static boolean gcd(long n, long m) {//其函数为求最大公约数，当公约数为1的时候，则其互质
        // TODO Auto-generated method stub
        long t=0;
        while(m>0) {
            t=n%m;
            n=m;
            m=t;//当=0说明两个数之间存在倍数关系
        }
        if(n==1)return true;
        return false;
    }
    public static void mainfind(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int t = scanner.nextInt();
        while (t-- > 0) {
            long n = scanner.nextLong();
            long k = scanner.nextLong();
            boolean flag = false;
            //三个数有公共的因数k，计算n/k，然后枚举出三数和等于n/k，且三个互质，不相等。结果就是三个数分别乘以k
            if ( n % k != 0) {
                System.out.println(-1);
                continue;
            }
            long p = n / k;

            for (long x = 1; x <= p - 3 && !flag; x++) {
                for (long y = x + 1; y < p - x && !flag; y++) {
                    long z =  (p - x - y);
                    if (z != x && z != y) {
                        if (!gcd(z, x)) continue;
                        if (!gcd(z, y)) continue;
                        if (!gcd(x, y)) continue;
                        System.out.println(x* k+ " " + y * k + " " + z * k);
                        flag = true;
                    }
                }
            }
            if (!flag) System.out.println(-1);
        }
        scanner.close();
    }
}
