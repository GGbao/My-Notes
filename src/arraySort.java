import java.util.Arrays;

public class arraySort {
    // 工具：交换数组中元素的位置
    public static int[] swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
        return arr;
    }
    // ****** 1.直接插入排序 ******

    /*直接插入排序是一种简单直观的排序方法。主要思想是对于未排序的元素，在已排序的元素中从后向前扫描，找到合适的位置后插入
    直接插入排序是稳定的。因为未排序的元素在向前扫描的过程中遇到相同的元素就不会继续向前扫描了，更不会插在它的前面。*/
    public static int[] InsertionSort(int[] arr){
        if (arr.length == 0 || arr.length == 1) {
            return arr;
        }
        for (int i = 0; i < arr.length-1; i++) {
            // 将 i+1 位置的数插入 0 到 i 之间的数组，从后往前遍历
            // current 指 i+1 的位置元素，pre 指 0 到 i 中依次向前遍历的指针
            int cur = arr[i + 1];
            int pre = i;
            while (pre >= 0 && cur < arr[pre]) {
                arr[pre + 1] = arr[pre];
                pre--;
            }
            // 最后将原来 i+1 位置的元素放入现在 0 到 i+1 之间数组中正确的位置上
            // pre+1 是因为刚才循环结束时又自减了一次
            arr[pre+1] = cur;
        }
        return arr;
    }
    // ****** 2.希尔排序 ******

    /*希尔排序是直接插入排序的升级版。主要思想是把一组数组分成了不同的“组”，只有同组元素才能比较和排序。
     随着排序的进行，“组”会慢慢减少，“组”中的元素也会慢慢增多，数组整体也会慢慢有序。
    希尔排序是不稳定的。虽然是否稳定是由该算法代码的具体实现决定的，但这种元素间远距离的交换一般都很难保证相同元素的相对位置
     希尔排序最重要的变量就是 gap，所有需要+1或者自加1的地方都要注意*/
    public static int[] ShellSort(int[] arr){
        if(arr.length == 0 || arr.length == 1)
            return arr;
        int current, gap = arr.length / 2;
        while(gap > 0){
            for (int i = gap; i < arr.length; i++) {
                // 从pre为0开始将 pre+gap 位置的数插入 0 到 pre 之间“同组”的数组，从后往前遍历
                // current 指 pre+gap 的位置元素
                current = arr[i];
                int pre = i - gap;
                while (pre >= 0 && current < arr[pre]) {
                    arr[pre + gap] = arr[pre];
                    pre -= gap;
                }
                arr[pre + gap] = current;
        }
            gap /= 2;
        }
        return arr;
    }

    //************简单选择排序*************
    /*
    简单选择排序是时间复杂度上最稳定的排序算法之一。排序方法很简单，每次都从未排序的数组中找到最小的元素，然后放在最前端。
    简单选择排序是不稳定的。毕竟它每趟只是选择最小的元素，选哪个可不一定，没办法保证两个相同元素的相对位置。
    * */
    // ****** 3.简单选择排序 ******
    public static int[] SelectionSort(int[] arr){
        if(arr.length == 0 || arr.length == 1)
            return arr;
        for(int i = 0;i < arr.length - 1;i++){
            // 每一轮挑出一个最小的元素，依次与不断增长的 i 位置的元素交换
            int MinIndex = i;
            for(int j = i;j < arr.length;j++){
                if(arr[j] < arr[MinIndex])
                    MinIndex = j;
            }
            arr = swap(arr,MinIndex,i);
            // 打印这一轮的排序结果
            System.out.println(Arrays.toString(arr));
        }
        return arr;
    }

    public static void main(String[] args) {
        int[] arr = {1, 11, 5, 6, 2, 9};
        System.out.println(Arrays.toString(ShellSort(arr)));

    }
}
