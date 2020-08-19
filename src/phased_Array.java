import java.math.BigDecimal;
import java.util.*;
import static java.lang.Math.*;
public class phased_Array {
    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
        //定义n维矩阵
        int n = 7;
        int x = 0;
        int y = 0;
        int z = 10;

        double[][] arr = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                arr[i][j] = calculateDelay(n, i, j, x, y, z);

            }
        }
    }
    private static double calculateDelay(int n, int i, int j, int x, int y, int z) {
        //单周期时间us
        double period = 1.0 / 24;//us

        //中心坐标
        int centreX = n / 2;
        int centreY = n / 2;
        //各点相对中心（0,0）坐标
        int distanceX =  j-centreX ;
        int distanceY = centreY - i;
        //x2+y2+z2开根号
        double totalDis = sqrt(pow(distanceX - x, 2) + pow(distanceY - y, 2) + pow(z, 2));
        double delay = (totalDis-z)*1000 / 34;//us
        double count = new BigDecimal(delay / period)
                .setScale(0, BigDecimal.ROUND_HALF_UP)
                .doubleValue();
        System.out.println("["+distanceX+","+distanceY+"]"+"--"+count);
        return 0;

    }
}
