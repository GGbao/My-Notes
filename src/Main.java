import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int t = sc.nextInt();
        int sum = 0;
        int res=0;
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = sc.nextInt();
            sum += a[i];
        }
        int target = sum / t;
        Arrays.sort(a);
        int left = 0, right = 0;
        for (int i = 0; i < n; i++) {
            if (a[i] == target) {
                left = i-1 ;
                break;
            }
        }
        for (int i = n-1; i >=0; i--) {
            if (a[i] == target) {
                right = i+1 ;
                break;
            }
        }
        int sum_l=0;
        int sum_r=0;
        for (int i = 0; i <= left; i++) {
            sum_l = sum_l + (target - a[i]);
        }
        for (int i = right; i <t; i++) {
            sum_r = sum_r + (a[i] - target + 1);
        }
        if (sum_r >= sum_l) {
            res = sum_l * 2;
        }



        System.out.println(res);

    }

}
