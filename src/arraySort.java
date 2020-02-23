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
    public static int[] InsertionSort(int[] arr) {
        if (arr.length == 0 || arr.length == 1) {
            return arr;
        }
        for (int i = 0; i < arr.length - 1; i++) {
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
            arr[pre + 1] = cur;
        }
        return arr;
    }
    // ****** 2.希尔排序 ******

    /*希尔排序是直接插入排序的升级版。主要思想是把一组数组分成了不同的“组”，只有同组元素才能比较和排序。
     随着排序的进行，“组”会慢慢减少，“组”中的元素也会慢慢增多，数组整体也会慢慢有序。
    希尔排序是不稳定的。虽然是否稳定是由该算法代码的具体实现决定的，但这种元素间远距离的交换一般都很难保证相同元素的相对位置
     希尔排序最重要的变量就是 gap，所有需要+1或者自加1的地方都要注意*/
    public static int[] ShellSort(int[] arr) {
        if (arr.length == 0 || arr.length == 1)
            return arr;
        int current, gap = arr.length / 2;
        while (gap > 0) {
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
    任何情况下 T(n) = O(n2)，所以说它在时间复杂度上稳定嘛。因为无论数组有序或是无序，简单选择排序都会遍历 n 遍这个数组。
    * */
    // ****** 3.简单选择排序 ******
    public static int[] SelectionSort(int[] arr) {
        if (arr.length == 0 || arr.length == 1)
            return arr;
        for (int i = 0; i < arr.length - 1; i++) {
            // 每一轮挑出一个最小的元素，依次与不断增长的 i 位置的元素交换
            int MinIndex = i;
            for (int j = i+1; j < arr.length; j++) {
                if (arr[j] < arr[MinIndex])
                    MinIndex = j;
            }
            arr = swap(arr, MinIndex, i);
            // 打印这一轮的排序结果
            System.out.println(Arrays.toString(arr));
        }
        return arr;
    }

    //*******************堆排序****************************
    /*
    * 堆排序是利用了最大堆这种数据结构的排序方法。因为每次将堆调整为最大堆后，
    * 堆的根结点一定是堆中最大的元素。我们通过不停的取出最大堆的根节点和重新调整为最大堆，就可以一直得到未排序数组的最大值。
    * 堆排序是不稳定的。毕竟远距离元素交换，不好保证相同元素的相对位置。任何情况下 T(n) = O(nlogn)。
    * - 二叉树中，叶节点为n-1时，父节点为n/2-1
    * 思路：a.将无需序列构建成一个堆，根据升序降序需求选择大顶堆或小顶堆;
　　        b.将堆顶元素与末尾元素交换，将最大元素"沉"到数组末端;
　　        c.重新调整结构，使其满足堆定义，然后继续交换堆顶元素与当前末尾元素，反复执行调整+交换步骤，直到整个序列有序
    * */
    // ****** 4.堆排序 ******
    // 主函数
    public static int[] HeapSort(int[] arr) {
        if (arr.length == 0 || arr.length == 1)
            return arr;
        //1、构建大顶堆,从最后一个非叶子结点开始,是从下而上，自右往左的调整
        for (int i = arr.length / 2 - 1; i >=0; i--) {
            adjustHeap(arr, i, arr.length);
        }
        //2、交换首尾项，去除末项重新构建大顶堆
        for (int i = arr.length - 1; i > 0; i--) {
            swap(arr, 0, i);
            adjustHeap(arr, 0, i);
        }
        return arr;
    }

    /**
     * 调整大顶堆作用
     *
     * @param arr
     * @param i      传入0即是交换完后的调整阶段
     * @param length
     */
    private static void adjustHeap(int[] arr, int i, int length) {
        int temp = arr[i];//先取出当前元素
        //叶子节点从左往右
        for (int k = i * 2 + 1; k < length; k = k * 2 + 1) {//从i结点的左子结点开始，也就是2i+1处开始
            if (k + 1 < length && arr[k + 1] > arr[k]) {//如果左子结点小于右子结点，k指向右子结点
                k++;
            }
            if (arr[k] > temp) {//如果子节点大于父节点，将子节点值赋给父节点（不用进行交换），i作为下次需要比较调整的坐标
                arr[i] = arr[k];
                i = k;
            } else {
                break;
            }
        }
        arr[i] = temp;//将temp值放到最终的位置。比较完后i即为最后所需要待的位置
    }

    // ****** 5.冒泡排序 ********
    //平均和最差情况 T(n) = O(n2)，最佳情况 T(n) = O(n)。

    public static int[] BubbleSort(int[] arr){
        for (int i = arr.length - 1; i > 0; i--) {
            for (int j = 0; j < i; j++) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr, j, j + 1);
                }
            }
        }
        return arr;

    }

    public static void main(String[] args) {
        int[] arr = {1, 11, 5, 6, 2, 9};
        System.out.println(Arrays.toString(SelectionSort(arr)));

    }
}
