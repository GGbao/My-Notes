import com.sun.org.apache.bcel.internal.generic.IF_ACMPEQ;

import javax.swing.tree.TreeNode;
import java.util.*;


public class TestCode {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    /*
    1、两数之和
    给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
    你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

    给定 nums = [2, 7, 11, 15], target = 9

    因为 nums[0] + nums[1] = 2 + 7 = 9
    所以返回 [0, 1]

    */

    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> tracker = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            if (!tracker.containsKey(nums[i])) {
                tracker.put(target - nums[i], i);
            } else {
                int left = tracker.get(nums[i]);
                return new int[]{left, i};
            }
        }
        return new int[2];
    }

    /**
     * 2、两数之和
     * 给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
     * <p>
     * 如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
     * <p>
     * 您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
     * <p>
     * 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
     * 输出：7 -> 0 -> 8
     * 原因：342 + 465 = 807
     */
    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(0);
        ListNode cur = pre;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int x = l1 == null ? 0 : l1.val;
            int y = l2 == null ? 0 : l2.val;
            int sum = x + y + carry;
            carry = sum / 10;
            sum = sum % 10;
            cur.next = new ListNode(sum);
            cur = cur.next;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry == 1) {
            cur.next = new ListNode(carry);
        }
        return pre;
    }

    /*
    3. 无重复字符的最长子串(滑块窗口)

    给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
    * */
    public int lengthOfLongestSubstring(String s) {
        int n = s.length(), max = 0, left = 0;
        Map<Character, Integer> map = new HashMap<>();
        for (int end = 0; end < n; end++) {
            if (map.containsKey(s.charAt(end))) {
                left = Math.max(left, map.get(s.charAt(end)));
            }
            map.put(s.charAt(end), end + 1);
            max = Math.max(end - left + 1, max);
        }
        return max;
    }

    /*
    4. 寻找两个有序数组的中位数

    给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。

    请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

    你可以假设 nums1 和 nums2 不会同时为空。

    * */

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        int[] t = new int[m + n];

        if (nums1.length == 0) {
            return getMiddle(nums2);
        }
        if (nums2.length == 0) {
            return getMiddle(nums1);
        }
        for (int i = 0, j = 0; i < nums1.length && j < nums2.length; ) {
            if (nums1[i] < nums2[j]) {
                t[i + j] = nums1[i];
                i++;
            } else {
                t[i + j] = nums2[j];
                j++;
            }
            if (i == nums1.length && j != nums2.length) {
                for (int k = i + j; k < m + n; k++) {
                    t[k] = nums2[j];
                    j++;

                }
            }
            if (j == nums2.length && i != nums1.length) {
                for (int k = i + j; k < m + n; k++) {
                    t[k] = nums1[i];
                    i++;

                }
            }


        }
        return getMiddle(t);

    }

    public double getMiddle(int[] num) {
        double mid;
        if (num.length % 2 == 0) {
            mid = (num[(num.length - 1) / 2] + num[(num.length + 1) / 2]) / 2;
        } else {
            mid = num[(num.length - 1) / 2];
        }
        return mid;
    }


    /*
    5. 最长回文子串

    * 给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
    * 输入: "babad"
    输出: "bab"
    注意: "aba" 也是一个有效答案。

    * */
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) {
            return "";
        }
        int start = 0;
        int end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);

    }

    public int expandAroundCenter(String s, int left, int right) {
        int l = left;
        int r = right;
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
            l--;
            r++;
        }
        return r - l - 1;
    }

    public String longPalindrome(String s) {
        int n = s.length();
        if (n < 2) {
            return s;
        }
        int maxlen = 1;
        String res = s.substring(0, 1);
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (j - i + 1 > maxlen && vaild(s, i, j)) {
                    maxlen = j - i + 1;
                    res = s.substring(i, j + 1);//substring 后一个索引得+1
                }
            }
        }
        return res;

    }

    public Boolean vaild(String s, int left, int right) {
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;

    }


   /* public class Solution {
        public double findMedianSortedArrays(int[] nums1, int[] nums2) {
            int[] nums = new int[1000];
            int t ,m= 0;
            for (int i = 0; i < nums1.length; i++) {
                nums[t] = nums1[i];
                t++;
            }
            for (int j = 0; j < nums2.length; j++) {
                nums[t] = nums2[j];
                t++;
            }
            if ((m = nums.length % 2) == 0) {
                return (nums[m] + nums[m + 1]) / 2;
            } else {
                return nums[m];
            }
        }
    }*/

    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s;
        }
        int len = Math.min(s.length(), numRows);
        String[] rows = new String[len];
        for (int i = 0; i < len; i++) {
            rows[i] = "";
        }
        int loc = 0;
        boolean down = false;
        for (int i = 0; i < s.length(); i++) {
            rows[loc] += s.substring(i, i + 1);
            if (loc == 0 || loc == numRows - 1) {

                down = !down;
                loc += down ? 1 : -1;
            }
        }
        String ans = "";
        for (String row : rows) {
            ans += row;
        }
        return ans;
    }

    /*
    7. 整数反转

    * 给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。


     * */
    public int reverse(int x) {
        int ans = 0;

        while (x != 0) {
            int pop = x % 10;
            if (ans > Integer.MAX_VALUE / 10 || (ans == Integer.MAX_VALUE / 10 && pop > 7)) {
                return 0;
            }
            if (ans < Integer.MIN_VALUE / 10 || (ans == Integer.MIN_VALUE / 10 && pop < -8)) {
                return 0;
            }
            ans = ans * 10 + pop;
            x = x / 10;
        }
        return ans;
    }

    public boolean isPalindrome(int x) {
        if (x < 0) {
            return false;
        }
        String s = String.valueOf(x);
        int n = s.length();
        if (n % 2 == 0) {
            int left = n / 2 - 1;
            int right = left + 1;
            return aroundCenter(s, left, right);
        } else {
            int left = n / 2;
            return aroundCenter(s, left, left);
        }
    }

    private Boolean aroundCenter(String s, int left, int right) {
        int l = left;
        int r = right;
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
            l--;
            r++;
        }
        return r - l - 1 == s.length();
    }

    /*
    11. 盛最多水的容器

    * 给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。
    * 在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。
    * 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

    说明：你不能倾斜容器，且 n 的值至少为 2。

    * */
    public int maxArea(int[] height) {

        int max = 0;
        int l = 0;
        int r = height.length - 1;
        while (l < r) {
            max = Math.max(max, Math.min(height[l], height[r]) * (r - l));
            if (height[l] < height[r]) {
                l++;
            } else {
                r--;
            }
        }
        return max;

    }


    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) {
            return "";
        }
        String ans = strs[0];
        int n = strs.length;
        for (int i = 1; i < n; i++) {
            int j = 0;
            for (; j < ans.length() && j < strs[i].length(); j++) {
                if (ans.charAt(j) != strs[i].charAt(j)) {
                    break;
                }
            }
            ans = ans.substring(0, j);
        }
        return ans;
    }

    /*
    15. 三数之和

    * 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

    注意：答案中不可以包含重复的三元组。

    * */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        int len = nums.length;
        if (len < 3 || nums == null) {

            return ans;
        }
        Arrays.sort(nums);
        for (int i = 0; i < len; i++) {
            if (nums[i] > 0) {// 如果当前数字大于0，则三数之和一定大于0，所以结束循环
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {// 去重
                continue;
            }
            int l = i + 1;
            int r = len - 1;
            while (r > l) {
                int sum = nums[i] + nums[l] + nums[r];

                if (sum == 0) {
                    ans.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while (r > l && nums[l] == nums[l + 1]) {
                        l++;// 去重
                    }
                    while (r > l && nums[r] == nums[r - 1]) {
                        r--;// 去重
                    }
                    l++;
                    r--;
                } else if (sum < 0) {
                    l++;
                } else if (sum > 0) {
                    r--;
                }

            }

        }
        return ans;


    }

    /*
    16. 最接近的三数之和

    * 给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，
    * 使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

    * */
    public int threeSumClosest(int[] nums, int target) {
        int ans = nums[0] + nums[1] + nums[2];
        int n = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i < n; i++) {
            int l = i + 1;
            int r = n - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (Math.abs(ans - target) > Math.abs(sum - target)) {
                    ans = sum;
                } else if (sum == target) {
                    return sum;
                } else if (sum > target) {
                    r--;
                } else if (sum < target) {
                    l++;
                }

            }

        }
        return ans;

    }

    /*17. 电话号码的字母组合


    * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。

    给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
    * */

    List<String> result = new ArrayList<>();
    char[][] chars = new char[10][];
    int[] int_input;

    public List<String> letterCombinations(String digits) {
        if (digits == null || "".equals(digits)) {
            return result;
        }
        chars[2] = new char[]{'a', 'b', 'c'};
        chars[3] = new char[]{'d', 'e', 'f'};
        chars[4] = new char[]{'g', 'h', 'i'};
        chars[5] = new char[]{'j', 'k', 'l'};
        chars[6] = new char[]{'m', 'n', 'o'};
        chars[7] = new char[]{'p', 'q', 'r', 's'};
        chars[8] = new char[]{'t', 'u', 'v'};
        chars[9] = new char[]{'w', 'x', 'y', 'z'};

        //输入
        char[] input = digits.toCharArray();
        int_input = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            int_input[i] = input[i] - 48;
        }
        //开始遍历
        for (int i = 0; i < chars[int_input[0]].length; i++) {
            dfs(chars[int_input[0]][i] + "", 1);
        }
        return result;

    }

    private void dfs(String preStr, int level) {
        //终止条件
        if (level == int_input.length) {
            result.add(preStr);
        } else {
            char[] chars_temp = chars[int_input[level]];
            for (char c : chars_temp) {
                dfs(preStr + c, level + 1);
            }
        }
    }

    /*
    18. 四数之和
    给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，
    使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。


    * */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        int len = nums.length;
        int temp;
        if (nums == null || len < 4) {
            return ans;
        }
        Arrays.sort(nums);

        for (int i = 0; i < len; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            temp = target - nums[i];
            for (int j = i + 1; j < len; j++) {
                // 去除j可能重复的情况
                if (j != i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                int l = j + 1;
                int r = len - 1;
                while (l < r) {
                    int sum = nums[j] + nums[l] + nums[r];
                    if (sum == temp) {
                        ans.add(Arrays.asList(nums[i], nums[j], nums[l], nums[r]));
                        while (l < r && nums[l] == nums[l + 1]) {
                            l++;
                        }
                        while (l < r && nums[r] == nums[r - 1]) {
                            r--;
                        }
                        l++;
                        r--;
                    } else if (sum > temp) {
                        r--;
                    } else if (sum < temp) {
                        l++;
                    }
                }
            }
        }
        return ans;

    }

    /**
     * 19. 删除链表的倒数第N个节点
     * 给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。
     * Definition for singly-linked list.
     * public class ListNode {
     * int val;
     * ListNode next;
     * ListNode(int x) { val = x; }
     * }
     */

    public ListNode removeNthFromEnd(ListNode head, int n) {
        int length = 0;
        int curnum = 0;
        ListNode node = head;
        while (node != null) {
            length++;
            node = node.next;
        }
        curnum = length - n;

        ListNode curNode = new ListNode(0);
        curNode.next = head;
        node = curNode;
        while (curnum > 0) {
            curnum--;
            node = node.next;

        }
        node.next = node.next.next;

        return curNode.next;
    }

    /*
    20. 有效的括号
    * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
    有效字符串需满足：
    左括号必须用相同类型的右括号闭合。
    左括号必须以正确的顺序闭合。
    注意空字符串可被认为是有效字符串。
*/
    public boolean isValid(String s) {
        if (s.length() % 2 != 0) {
            return false;
        }
        if (s.length() == 0) {
            return true;
        }
        int index = 0;
        char[] stock = new char[s.length()];
        for (int i = 0; i < s.length(); i++) {
            switch (s.charAt(i)) {
                case '(':
                case '[':
                case '{':
                    stock[index++] = s.charAt(i);
                    continue;
                case ')':
                    if (index == 0 || stock[--index] != '(') {
                        return false;
                    }
                    continue;
                case ']':
                    if (index == 0 || stock[--index] != '[') {
                        return false;
                    }
                    continue;
                case '}':
                    if (index == 0 || stock[--index] != '{') {
                        return false;
                    }
                    continue;
            }
        }
        return index == 0;


    }

    /*
    21. 合并两个有序链表
    * 将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
    * */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l2.next, l1);
            return l2;
        }


    }

    /*
    22. 括号生成


    * 给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
    * */

    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        //排除特殊情况
        if (n == 0) {
            return res;
        }
        //深度优先搜索遍历
        dfs("", n, n, res);
        return res;
    }

    /**
     * @param curStr 当前递归得到的结果
     * @param left   左括号剩余个数
     * @param right  右括号剩余个数
     * @param res    结果集
     */
    private void dfs(String curStr, int left, int right, List<String> res) {
        if (left == 0 && right == 0) {
            res.add(curStr);
            return;
        }
        //剪枝
        if (left > right) {
            return;
        }
        if (left > 0) {
            dfs(curStr + "(", left - 1, right, res);
        }
        if (right > 0) {
            dfs(curStr + ")", left, right - 1, res);
        }
    }

    /*
    * 24. 两两交换链表中的节点
    * 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
    *
    你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

    示例:
    给定 1->2->3->4, 你应该返回 2->1->4->3.

    * */
    public ListNode swapPairs(ListNode head) {

        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode cur = pre;

        while (cur.next != null && cur.next.next != null) {
            ListNode start = cur.next;
            ListNode end = start.next;
            cur.next = end;
            start.next = end.next;
            end.next = start;
            cur = start;

        }
        return pre.next;
    }

    /*
    * 31. 下一个排列
实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

     * */
    public void nextPermutation(int[] nums) {
        if (nums.length <= 1) {
            return;
        }
        //1、找出相邻升序(i,j)
        int i = nums.length - 2;
        int j = nums.length - 1;
        int k = nums.length - 1;

        for (; i >= 0 && nums[i] >= nums[j]; ) {
            i--;
            j--;
        }
        if (j <= 0) {
            for (int x = nums.length - 1; x >= j; x--, j++) {
                int res = nums[x];
                nums[x] = nums[j];
                nums[j] = res;
            }
            return;
        }
        //2、(j,end)从后往前查询a[i]<a[k],swap
        for (; k >= j && nums[i] >= nums[k]; ) {
            k--;
        }
        int temp = nums[k];
        nums[k] = nums[i];
        nums[i] = temp;
        //3、逆置[j,end)
        for (int l = nums.length - 1; l >= j; l--, j++) {
            int res = nums[l];
            nums[l] = nums[j];
            nums[j] = res;
        }
    }

    /*
    * 33. 搜索旋转排序数组
假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 O(log n) 级别。
*
* 二分中的mid最好写成start+(end-start)/2


     * */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int start = 0;
        int end = nums.length - 1;
        int mid;
        while (start <= end) {
            mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] >= nums[end]) {
                //前半部分有序
                if (target >= nums[start] && target <= nums[mid]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }

            } else {
                //后半段有序
                if (target >= nums[mid] && target <= nums[end]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }
        }
        return -1;


    }


    /*
    * 34. 在排序数组中查找元素的第一个和最后一个位置
    给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

    你的算法时间复杂度必须是 O(log n) 级别。

    如果数组中不存在目标值，返回 [-1, -1]。

     * */
    public int[] searchRange(int[] nums, int target) {
        if (nums == null && nums.length == 0) {
            return new int[]{1, -1};
        }
        if (nums.length == 1 && nums[0] == target) {
            return new int[]{0, 0};
        }
        int start = 0;
        int end = nums.length - 1;
        int mid;
        int cur;
        int left = -1;
        int right = -1;

        while (start <= end) {
            mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                left = right = mid;
                cur = mid;
                while (cur > 0) {
                    if (nums[cur - 1] == nums[cur]) {
                        left = cur - 1;
                        cur = cur - 1;
                    } else {
                        break;
                    }

                }
                cur = mid;
                while (cur < nums.length - 1) {
                    if (nums[cur + 1] == nums[cur]) {
                        right = cur + 1;
                        cur++;
                    } else {
                        break;
                    }
                }
                return new int[]{left, right};
            } else if (nums[mid] > target) {
                //在前面
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        return new int[]{left, right};

    }

    /*
    * 35. 搜索插入位置
    给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
    你可以假设数组中无重复元素。

     * */
    public int searchInsert(int[] nums, int target) {
        if (target < nums[0]) {
            return 0;
        }
        if (target > nums[nums.length - 1]) {
            return nums.length;
        }
        int left = 0;
        int right = nums.length - 1;
        int mid;
        while (left <= right) {
            mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        return left;

    }

    /*
    * 36. 有效的数独
    判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。

     * */
    public boolean isValidSudoku(char[][] board) {
        HashMap<Integer, Integer>[] rosw = new HashMap[9];
        HashMap<Integer, Integer>[] columns = new HashMap[9];
        HashMap<Integer, Integer>[] boxes = new HashMap[9];
        for (int i = 0; i < 9; i++) {
            rosw[i] = new HashMap<Integer, Integer>();
            columns[i] = new HashMap<Integer, Integer>();
            boxes[i] = new HashMap<Integer, Integer>();
        }
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int n = board[i][j];
                    int box_index = (i / 3) * 3 + (j / 3);

                    rosw[i].put(n, rosw[i].getOrDefault(n, 0) + 1);
                    columns[j].put(n, columns[j].getOrDefault(n, 0) + 1);
                    boxes[box_index].put(n, boxes[box_index].getOrDefault(n, 0) + 1);

                    if (rosw[i].get(n) > 1 || columns[j].get(n) > 1 || boxes[box_index].get(n) > 1) {
                        return false;
                    }
                }
            }
        }
        return true;

    }

    /*
    * 38. 外观数列
    * 「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。前五项如下：
    1.     1
    2.     11
    3.     21
    4.     1211
    5.     111221
    * */
    public String countAndSay(int n) {
        String str = "1";
        for (int i = 2; i <= n; i++) {
            StringBuilder builder = new StringBuilder();
            char pre = str.charAt(0);
            int count = 1;
            for (int j = 1; j < str.length(); j++) {
                char c = str.charAt(j);
                if (c == pre) {
                    count++;
                } else {
                    builder.append(count).append(pre);
                    pre = c;
                    count = 1;
                }
            }
            builder.append(count).append(pre);
            str = builder.toString();
        }
        return str;
    }


    /*
    * 39. 组合总和
    * 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
    candidates 中的数字可以无限制重复被选取。

    说明：

    所有数字（包括 target）都是正整数。
    解集不能包含重复的组合。 
    输入: candidates = [2,3,6,7], target = 7,
    所求解集为:
    [
    [7],
    [2,2,3]
    ]
     * */
    List<List<Integer>> res = new ArrayList<>();
    LinkedList<Integer> track = new LinkedList<>();
    int[] candidates;
    int len;

    public List<List<Integer>> combinationSum(int[] candidates, int target) {

        Arrays.sort(candidates);
        this.candidates = candidates;
        this.len = candidates.length;

        findCombinationSum(target, 0, track);
        return res;

    }

    private void findCombinationSum(int residue, int start, LinkedList<Integer> track) {
        if (residue == 0) {
            res.add(new ArrayList<>(track));
            return;
        }
        for (int i = start; i < len && residue - candidates[i] >= 0; i++) {
            track.add(candidates[i]);
            findCombinationSum(residue - candidates[i], i, track);
            track.removeLast();

        }

    }

    /*
    * 40. 组合总和 II
    * 给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
    candidates 中的每个数字在每个组合中只能使用一次。
    说明：
    所有数字（包括目标数）都是正整数。
    解集不能包含重复的组合。 

     * */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        this.candidates = candidates;
        this.len = candidates.length;

        findCombinationSum(target, 0, track);
        return res;

    }

    private void findCombinationSum2(int residue, int start, LinkedList<Integer> track) {
        if (residue == 0) {
            res.add(new ArrayList<>(track));
            return;
        }
        for (int i = start; i < len && residue - candidates[i] >= 0; i++) {
            //在一个for循环中，所有被遍历到的数都是属于一个层级的。
            // 我们要让一个层级中，必须出现且只出现一个2，那么就放过第一个出现重复的2，但不放过后面出现的2。
            // 第一个出现的2的特点就是 i==start. 第二个出现的2 特点是i > start
            if (i > start && candidates[i] == candidates[i + 1]) {
                continue;
            }
            track.add(candidates[i]);
            findCombinationSum(residue - candidates[i], i + 1, track);
            track.removeLast();

        }

    }

    /*
    *43. 字符串相乘
    * 给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
    示例 1:

    输入: num1 = "2", num2 = "3"
    输出: "6"


    * */
    public String multiply(String num1, String num2) {
        if ("0".equals(num1) || "0".equals(num2)) {
            return "0";

        }
        int lenth1 = num1.length();
        int lenth2 = num2.length();
        StringBuilder builder = new StringBuilder();
        int[] resInt = new int[lenth1 + lenth2];
        for (int i = lenth1 - 1; i >= 0; i--) {
            for (int j = lenth2 - 1; j >= 0; j--) {
                int number1 = num1.charAt(i) - '0';
                int number2 = num2.charAt(j) - '0';
                resInt[i + j] += number1 * number2;
                if (resInt[i + j] >= 10 && (i + j) != 0) {
                    resInt[i + j - 1] += resInt[i + j] / 10;
                    resInt[i + j] = resInt[i + j] % 10;
                }
            }
        }
        for (int i = 0; i <= resInt.length - 2; i++) {
            builder.append(resInt[i] + "");
        }

        return builder.toString();


    }

    /*
     * 46. 全排列
     * 给定一个没有重复数字的序列，返回其所有可能的全排列。

     * */
    public List<List<Integer>> permute(int[] nums) {
        this.len = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        LinkedList<Integer> track = new LinkedList<>();
        Arrays.sort(nums);

        def46(res, nums, track);
        return res;

    }

    private void def46(List<List<Integer>> res, int[] nums, LinkedList<Integer> stack) {
        if (stack.size() == len) {
            res.add(new ArrayList<>(stack));
            return;
        }
        for (int j = 0; j < nums.length; j++) {
            if (stack.contains(nums[j])) {
                continue;
            }
            stack.add(nums[j]);
            def46(res, nums, stack);
            stack.removeLast();

        }

    }

    /*
     * 47. 全排列 II
     * 给定一个可包含重复数字的序列，返回所有不重复的全排列。


     * */
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        LinkedList<Integer> track = new LinkedList<>();
        boolean[] flag = new boolean[nums.length];
        Arrays.sort(nums);

        dfs47(nums, flag, track, res);
        return res;
    }

    private void dfs47(int[] nums, boolean[] flag, LinkedList<Integer> track, List<List<Integer>> res) {
        if (track.size() == nums.length) {
            res.add(new ArrayList<>(track));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (flag[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && flag[i - 1]) {
                continue;
            }
            track.add(nums[i]);
            flag[i] = true;
            dfs47(nums, flag, track, res);
            track.removeLast();
            flag[i] = false;
        }
    }

    /*
    * 48. 旋转图像
    * 给定一个 n × n 的二维矩阵表示一个图像。

    将图像顺时针旋转 90 度。

    说明：

    你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。
     * */
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        //转置，对角交换
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        //逐行逆转
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {

                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n - j - 1];
                matrix[i][n - j - 1] = temp;


            }
        }


    }

    /*
     * 49. 字母异位词分组
     * 给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
     *["eat", "tea", "tan", "ate", "nat", "bat"]
     * */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List> ans = new HashMap<String, List>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = String.valueOf(chars);
            if (!ans.containsKey(key)) {
                ans.put(key, new ArrayList());
            }
            ans.get(key).add(str);
        }
        return new ArrayList(ans.values());

    }

    /*
    53. 最大子序和
    给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
    * */
    public int maxSubArray(int[] nums) {
        int max = nums[0];
        for (int i = 0; i < nums.length; i++) {
            max = findmax(max, i, nums);
        }
        return max;

    }

    private int findmax(int max, int i, int[] nums) {
        int temp = 0;
        for (int j = i; j < nums.length; j++) {
            temp = temp + nums[j];
            if (temp >= max) {
                max = temp;

            }
        }
        return max;
    }

    /*
     * 54. 螺旋矩阵
     * 给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。
     * */
    public List<Integer> spiralOrder(int[][] matrix) {
        List ans = new ArrayList();
        if (matrix.length == 0) {
            return ans;
        }
        int R = matrix.length;
        int C = matrix[0].length;
        boolean[][] seen = new boolean[R][C];
        int[] dr = {0, 1, 0, -1};//游标移动方向控制
        int[] dc = {1, 0, -1, 0};
        int r = 0, c = 0, di = 0;
        for (int i = 0; i < R * C; i++) {
            ans.add(matrix[r][c]);
            seen[r][c] = true;
            int cr = r + dr[di];
            int cc = c + dc[di];
            if (cr >= 0 && cc >= 0 && cr < R && cc < C && !seen[cr][cc]) {
                r = cr;
                c = cc;
            } else {
                di = (di + 1) % 4;
                r += dr[di];
                c += dc[di];
            }

        }
        return ans;

    }

    /*
    * 55. 跳跃游戏
    * 给定一个非负整数数组，你最初位于数组的第一个位置。
    数组中的每个元素代表你在该位置可以跳跃的最大长度。
    判断你是否能够到达最后一个位置。
    * [2,3,1,1,4]
     * */
    public boolean canJump(int[] nums) {
        boolean[] dp = new boolean[nums.length];
        if (nums[0] == 0) {
            if (nums.length == 1) {
                return true;
            }
            return false;
        }
        dp[0] = true;

        for (int i = 1; i < nums.length; i++) {
            if (dp[nums.length - 1]) {
                break;
            } else if (dp[i - 1] && nums[i - 1] > 0) {
                dp[i] = true;
                for (int j = 1; j <= nums[i - 1] && i + j <= nums.length; j++) {
                    dp[i + j - 1] = true;
                }
            } else {
                dp[i] = false || dp[i];
            }

        }
        return dp[nums.length - 1];


    }

    /*
    * 56. 合并区间
    * 给出一个区间的集合，请合并所有重叠的区间。
    * 输入: [[1,3],[2,6],[8,10],[15,18]]
      输出: [[1,6],[8,10],[15,18]]
     * */
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return new int[0][];
        }
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });

        int len = intervals.length;
        LinkedList<int[]> res = new LinkedList<>();
        res.add(intervals[0]);
        for (int i = 1; i < len; i++) {
            int[] last = res.getLast();
            int left = intervals[i][0];
            int right = intervals[i][1];
            if (left > last[1]) {
                res.add(intervals[i]);
            } else if (left < last[0]) {
                last[0] = left;
            } else if (left <= last[1]) {
                last[1] = right;
            }
        }

        return res.toArray(new int[0][]);

    }

    /*
     * 59. 螺旋矩阵 II
     * 给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
     * */
    public int[][] generateMatrix(int n) {
        int[][] ans = new int[n][];
        int r1 = 0, r2 = n - 1;
        int c1 = 0, c2 = n - 1;
        int cur = 1;

        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; c++) {
                ans[r1][c] = cur++;
            }
            for (int r = r1 + 1; r <= r2; r++) {
                ans[r][c2] = cur++;
            }
            for (int c = c2 - 1; c >= c1; c--) {
                ans[r2][c] = cur++;
            }
            for (int r = r2 - 1; r >= r1 + 1; r--) {
                ans[r][c1] = cur++;
            }
            r1++;
            r2--;
            c1++;
            c2--;
        }
        return ans;
    }

    /*
    * 60. 第k个排列
    * 给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。
    按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：
     * */
    public String getPermutation(int n, int k) {
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            ans[i] = i + 1;
        }
        int cur = 1;

        while (cur < k) {

            int i, j, x;

            //逆序找到(j,end) n[i]<n[j]
            for (j = n - 1; j > 0; j--) {
                if (ans[j - 1] < ans[j]) {
                    break;
                }
            }
            i = j - 1;
            for (x = n - 1; x >= j; x--) {
                if (ans[x] >= ans[i]) {
                    int temp = ans[x];
                    ans[x] = ans[i];
                    ans[i] = temp;
                    break;
                }
            }
            //3、逆置[j,end)
            for (int l = n - 1; l >= j; l--, j++) {
                int res = ans[l];
                ans[l] = ans[j];
                ans[j] = res;
            }
            cur++;
        }
        StringBuilder builder = new StringBuilder();
        for (int i : ans) {
            builder.append(i);
        }
        return builder.toString();

    }

    /*
    * 62. 不同路径
    * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

    机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

    问总共有多少条不同的路径？

     * */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int j = 0; j < m; j++) {
            dp[j][0] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    /*
    * 63. 不同路径 II
    * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
    机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
    现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
    * */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][j] = 0;
            }
        }
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 1) {
                break;

            }
            dp[0][i] = 1;
        }
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1) {
                break;
            }
            dp[i][0] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    continue;
                }
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    /*
    * 64. 最小路径和
    * 给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
    说明：每次只能向下或者向右移动一步。
    * */
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int j = 1; j < m; j++) {
            dp[j][0] = dp[j - 1][0] + grid[j][0];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    /*
    * 66. 加一
    * 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
    最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
    你可以假设除了整数 0 之外，这个整数不会以零开头。
    * */
    public int[] plusOne(int[] digits) {
        int n = digits.length;
        for (int i = n - 1; i >= 0; i--) {
            digits[i]++;
            digits[i] %= 10;
            if (digits[i] != 0) {
                return digits;
            }
        }
        digits = new int[n + 1];
        digits[0] = 1;
        return digits;
    }

    /*
     * 67. 二进制求和
     *内置函数
     * */
    public String addBinary(String a, String b) {
        //将 a 和 b 转换为十进制整数。
        //求和。
        //将求和结果转换为二进制整数。
        return Integer.toBinaryString(Integer.parseInt(a, 2) + Integer.parseInt(b, 2));
    }

    /*
    * 69. x 的平方根
    计算并返回 x 的平方根，其中 x 是非负整数。
    由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
    * */

    //(1) 一个非负整数的算数平方根一定在 0 和它自身之间，可以使用二分查找。
    //(2) 大于等于 2 的算数平方根只保留整数部分都小于等于输入值的一半。
    //(3) 不断查找中间值平方后和输入值比较来找到保留整数部分的算术平方根。
    public int mySqrt(int x) {
        if (x == 0) {
            return 0;
        }
        if (x == 1) {
            return 1;
        }
        int l = 2;
        int r = x / 2;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            long sum = (long) mid * mid;
            if (sum > x) {
                r = mid - 1;
            } else if (sum < x) {
                l = mid + 1;
            } else {
                return mid;
            }
        }
        return r;
    }

    /*
    * 70. 爬楼梯
    * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
    * */
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    /*
    73. 矩阵置零
    给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用原地算法。
    * */
    public void setZeroes(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        Set<Integer> map_row = new HashSet();
        Set<Integer> map_column = new HashSet();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    map_row.add(i);
                    map_column.add(j);
                } else {
                    continue;
                }
            }
        }
        for (int row : map_row) {
            for (int i = 0; i < n; i++) {
                matrix[row][i] = 0;
            }
        }
        for (int column : map_column) {
            for (int i = 0; i < m; i++) {
                matrix[i][column] = 0;
            }
        }

    }

    /*
    *74. 搜索二维矩阵
        编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
        每行中的整数从左到右按升序排列。
        每行的第一个整数大于前一行的最后一个整数。
    * */
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int r = 0;
        int c = matrix[0].length - 1;
        while (r < matrix.length && c >= 0) {
            if (matrix[r][c] == target) {
                return true;
            } else if (matrix[r][c] > target) {
                c--;
            } else {
                r++;
            }
        }
        return false;
    }

    /*
     * 75. 颜色分类
     * 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列
     * 此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
     * */
    //快排
    public void sortColors(int[] nums) {
        int start = 0;
        int end = nums.length - 1;
        quickSort(nums, start, end);
    }

    private void quickSort(int[] nums, int start, int end) {
        if (start < end) {
            int i = adjustArray(nums, start, end);
            quickSort(nums, start, i - 1);
            quickSort(nums, i + 1, end);
        }
    }

    private int adjustArray(int[] nums, int l, int r) {
        int x = nums[l];
        while (l < r) {
            while (l < r && x <= nums[r]) {
                r--;
            }
            if (l < r) {
                nums[l++] = nums[r];
            }
            while (l < r && x >= nums[l]) {
                l++;
            }
            if (l < r) {
                nums[r--] = nums[l];
            }
        }
        nums[l] = x;
        return l;
    }

    /*
     * 77. 组合
     * 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
     * 回溯算法就是在一个树形问题上的深度优先遍历，所以先把这个问题对应的树的结构画出来,一定要画图！
     * */
    public List<List<Integer>> combine(int n, int k) {
        if (n <= 0 || k <= 0 || n < k) {
            return res;
        }
        findCombine(n, k, 1, new LinkedList<>());
        return res;
    }

    private void findCombine(int n, int k, int begin, LinkedList<Integer> pre) {
        if (pre.size() == k) {
            res.add(new ArrayList<>(pre));
            return;
        }
        for (int i = begin; i <= n; i++) {
            pre.add(i);
            findCombine(n, k, i + 1, pre);
            //剔除已经加入的元素，往下遍历
            pre.removeLast();
        }
    }

    /*
     * 78. 子集
     * 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
     * */
    public List<List<Integer>> subsets(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i <= n; i++) {
            findSub(n, i, nums, 0, new LinkedList<>());
        }
        return res;
    }

    private void findSub(int n, int i, int[] nums, int begin, LinkedList<Integer> pre) {
        if (pre.size() == i) {
            res.add(new ArrayList<>(pre));
            return;
        }
        for (int j = begin; j < n; j++) {
            pre.add(nums[j]);
            findSub(n, i, nums, j + 1, pre);
            pre.removeLast();
        }
    }

    /*
    *79. 单词搜索
        给定一个二维网格和一个单词，找出该单词是否存在于网格中。
        单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。
        同一个单元格内的字母不允许被重复使用。
    * */
    boolean[][] isVisited;
    int[][] direction = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    int m, n;

    public boolean exist(char[][] board, String word) {
        m = board.length;
        n = board[0].length;
        isVisited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfs79(i, j, 0, word, board)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * @param index 指针指向的索引
     */
    private boolean dfs79(int i, int j, int index, String word, char[][] board) {
        if (index == word.length() - 1) {
            return board[i][j] == word.charAt(index);
        }
        if (board[i][j] == word.charAt(index)) {
            isVisited[i][j] = true;
            for (int k = 0; k < 4; k++) {
                int x = i + direction[k][0];
                int y = j + direction[k][1];
                //在范围内在进行下一个坐标比较
                if (inArea(x, y) && !isVisited[x][y] && dfs79(x, y, index + 1, word, board)) {
                    return true;
                }
            }
            //四个方向都不对时就回溯
            isVisited[i][j] = false;
        }
        return false;
    }

    private boolean inArea(int x, int y) {
        return x >= 0 && y >= 0 && x < m && y < n;
    }

    /*
     * 26. 删除排序数组中的重复项
     * 给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
     * */
    //思路：考虑用 2 个指针，一个在前记作 p，一个在后记作 q，算法流程如下：
    //1.比较 p 和 q 位置的元素是否相等。
    //如果相等，q 后移 1 位
    //如果不相等，将 q 位置的元素复制到 p+1 位置上，p 后移一位，q 后移 1 位
    //重复上述过程，直到 q 等于数组长度。
    //返回 p + 1，即为新数组长度。

    public int removeDuplicates26(int[] nums) {
        int p = 0, q = 1;
        while (q < nums.length) {
            if (nums[p] != nums[q]) {
                nums[p + 1] = nums[q];
                p++;
            }
            q++;
        }
        return p + 1;
    }

    //对index变量进行修改:[0,index] 表示不重复元素集合
    public int removeDuplicates26_2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        // 1.指针定义 [0,index] 是修改后无重复的排序元素 注意 这里已经把0纳入进去了
        int index = 0;
        // 2.另一个循环指针 从1开始，终止为nums.length，为什么从1开始 因为我们要比较重复 nums[0] 肯定是不重复的
        for (int i = 1; i < nums.length; i++) {
            // 3.指针运动的条件
            if (nums[index] != nums[i]) {
                index++;
                nums[index] = nums[i];
            }
        }
        // 4.根据定义确定返回值
        return index + 1;
    }


    /*
    * 80. 删除排序数组中的重复项 II
    给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
    * */
    public int removeDuplicates(int[] nums) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                if (map.get(nums[i]) < 2) {
                    map.put(nums[i], map.get(nums[i]) + 1);
                }
            } else {
                map.put(nums[i], 1);
            }
        }
        int count = 0;
        int cur = 0;
        for (Integer num : map.keySet()) {
            count += map.get(num);
            for (; cur < count; cur++) {
                nums[cur] = num;
            }

        }

        return count;
    }

    public int removeDuplicates_1(int[] nums) {
        if (nums == null) {
            return 0;
        }
        if (nums.length <= 2) {
            return nums.length;
        }
        //定义[0,index]是修改后满足的数组区间
        //先把k-1个元素装入
        int index = 1;
        //判断中止条件,k开始
        for (int i = 2; i < nums.length; i++) {
            //指针移动(index-1)为重复区间的首个元素[index-k+1]
            if (nums[i] != nums[index - 1]) {
                index++;
                nums[index] = nums[i];
            }
        }
        return index + 1;
    }

    /*
     * 82. 删除排序链表中的重复元素 II
     * 给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。
     * */
    public ListNode deleteDuplicates(ListNode head) {
        ListNode pre = new ListNode(-1);
        pre.next = head;
        ListNode slow = pre;
        ListNode fast = pre.next;
        while (fast != null) {
            //判断快节点到重复之前
            while (fast.next != null && fast.val == fast.next.val) {
                fast = fast.next;
            }
            //没有重复，慢节点后移
            if (slow.next == fast) {
                slow = slow.next;
            } else {
                slow.next = fast.next;
            }
            //每次快节点都要后移
            fast = fast.next;
        }
        return pre.next;
    }

    /*
     * 83. 删除排序链表中的重复元素
     * 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
     * */
    public ListNode deleteDuplicates83(ListNode head) {
        ListNode pre = new ListNode(-1000);
        pre.next = head;
        ListNode slow = pre;
        ListNode fast = pre.next;
        while (slow.next != null) {
            if (slow.val != fast.val) {
                slow = fast;
                fast = fast.next;
            } else {
                fast = fast.next;
            }
            slow.next = fast;
        }
        return pre.next;

    }

    /*
     * 86. 分隔链表
     * 给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。
     * */
    public ListNode partition(ListNode head, int x) {
        ListNode pre1 = new ListNode(0);//记录小值链表的头
        ListNode minC = pre1;//对小表操作用的指针
        ListNode pre2 = new ListNode(0);//记录大值链表的头
        ListNode maxC = pre2;//对大表操作用的指针
        ListNode cur = head;
        while (cur != null) {
            if (cur.val < x) {
                minC.next = cur;//放入小值链表中，指针后移一位
                minC = cur;
            } else {
                maxC.next = cur;
                maxC = cur;
            }
            cur = cur.next;
        }
        //遍历完后一段链表的最后节点指向null
        maxC.next = null;
        minC.next = pre2.next;
        return pre1.next;
    }

    /*
     * 88. 合并两个有序数组
     * 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
     * */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int index = m + n - 1;
        if (m == 0) {
            for (int i = 0; i <= index; i++) {
                nums1[i] = nums2[i];
            }
        }

        while (n > 0 && m > 0 && index >= 0) {
            if (nums1[m - 1] <= nums2[n - 1]) {
                nums1[index--] = nums2[--n];
            } else {
                nums1[index--] = nums1[--m];
            }
        }
        while (n > 0) {
            nums1[index--] = nums2[--n];
        }

    }

    /*
    * 90. 子集 II
    * 给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
        说明：解集不能包含重复的子集。
    * */
//    Set<List<Integer>> set = new HashSet<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i <= n; i++) {
            findSub90(n, i, 0, nums, new LinkedList<>());
        }
        return res;
    }

    private void findSub90(int n, int i, int begin, int[] nums, LinkedList<Integer> pre) {
        if (pre.size() == i) {
            if (!res.contains(pre)) {
                res.add(new ArrayList<>(pre));
            }
            return;
        }
        for (int j = begin; j < n; j++) {
            pre.add(nums[j]);
            findSub90(n, i, j + 1, nums, pre);
            pre.removeLast();
        }
    }

    /*
     *91. 解码方法
     * 一条包含字母 A-Z 的消息通过以下方式进行了编码：
     * */
    //dp[]，用来记录以某个字符为开始的解码数
    //num[i]=0,dp[i]=0
    //如果num[i]+num[i+1]<=26,dp[i]=dp[i+1]+dp[i+2],否则dp[i]=dp[i+1];
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int len = s.length();
        int[] dp = new int[len + 1];
        //设置最后一位为1
        dp[len] = 1;
        if (s.charAt(len - 1) == '0') {
            dp[len - 1] = 0;
        } else {
            dp[len - 1] = 1;
        }
        for (int i = len - 2; i >= 0; i--) {
            if (s.charAt(i) == '0') {
                dp[i] = 0;
                continue;
            }
            if ((s.charAt(i) - '0') * 10 + (s.charAt(i + 1) - '0') <= 26) {
                dp[i] = dp[i + 1] + dp[i + 2];
            } else {
                dp[i] = dp[i + 1];
            }
        }
        return dp[0];
    }

    /*
     * 92. 反转链表 II
     * 反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
     * */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode pre = new ListNode(-1);
        pre.next = head;
        ListNode g = pre;
        int count = 1;

        while (count < m) {
            g = g.next;
            count++;
        }
        //g为指定节点之前的节点，cur为当前节点
        ListNode cur = g.next;
        for (int i = 0; i < n - m; i++) {
            ListNode removed = cur.next;
            cur.next = cur.next.next;
            removed.next = g.next;
            g.next = removed;
        }
        return pre.next;
    }

    /*
     * 93. 复原IP地址
     * 给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。
     * 有效的 IP 地址正好由四个整数（每个整数位于 0 到 255 之间组成），整数之间用 '.' 分隔。
     * */
    //splitTimes：已经分割出多少个 ip 段；
    //begin：截取 ip 段的起始位置；
    //path：记录从根结点到叶子结点的一个路径（回溯算法常规变量，是一个栈）；
    //res：记录结果集的变量，常规变量。

    public List<String> restoreIpAddresses(String s) {
        int n = s.length();
        List<String> res = new ArrayList<>();
        //如果长度不够，不搜索
        if (n < 4 || n > 12) {
            return res;
        }
        int splitTimes = 0;
        Deque<String> path = new ArrayDeque<>(4);
        dfs93(s, n, splitTimes, 0, path, res);
        return res;
    }

    private void dfs93(String s, int n, int splitTimes, int begin, Deque<String> path, List<String> res) {
        //中止条件
        if (begin == n) {
            if (splitTimes == 4) {
                res.add(String.join(".", path));
            }
            return;
        }
        //看剩下的不够了，就退出剪枝，n-begin表示剩余的还未分割的字符串位数
        if (n - begin < (4 - splitTimes) || n - begin > 3 * (4 - splitTimes)) {
            return;
        }
        for (int i = 0; i < 3; i++) {
            if (begin + i >= n) {
                break;
            }
            int ipSegment = isvalid(s, begin, begin + i);
            if (ipSegment != -1) {
                //再判断ip段有效时才去截取
                path.addLast(ipSegment + "");
                dfs93(s, n, splitTimes + 1, begin + i + 1, path, res);
                path.removeLast();
            }
        }

    }

    //判断s的子区间是否有效
    private int isvalid(String s, int left, int right) {
        int n = right - left + 1;
        // 大于 1 位的时候，不能以 0 开头
        if (n > 1 && s.charAt(left) == '0') {
            return -1;
        }
        //转成int类型
        int res = 0;
        for (int i = left; i <= right; i++) {
            res = res * 10 + s.charAt(i) - '0';
        }
        if (res > 255) {
            return -1;
        }
        return res;
    }

    /*
     * 94. 二叉树的中序遍历
     * 给定一个二叉树，返回它的中序 遍历。
     * */
    public List<Integer> inorderTraversal(TreeNode root) {
        //stack栈顶元素永远为cur父节点
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> list = new ArrayList<>();
        TreeNode cur = root;
        if (cur == null) {
            return list;
        }
        while (cur != null || !stack.isEmpty()) {
            //先到达最左节点
            if (cur != null) {
                stack.push(cur);
                cur = cur.left;
            } else {
                list.add(stack.peek().val);
                cur = stack.pop().right;
            }
        }
        return list;
    }

    /*
     * 95. 不同的二叉搜索树 II
     * 给定一个整数 n，生成所有由 1 ... n 为节点所组成的二叉搜索树
     * */
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> res = new ArrayList<>();
        if (n == 0) {
            return res;
        }
        return getRes(1, n);

    }

    private List<TreeNode> getRes(int start, int end) {
        List<TreeNode> res = new ArrayList<>();
        //此时没有数字，把null作为子树加入结果
        if (start > end) {
            res.add(null);
            return res;
        }
        //只有一个数组，当前数字作为一颗字数加入结果集
        if (start == end) {
            TreeNode tree = new TreeNode(start);
            res.add(tree);
            return res;
        }
        //尝试每个数字作为根节点
        for (int i = start; i <= end; i++) {
            //得到所有可能的左子树
            List<TreeNode> leftTrees = getRes(start, i - 1);
            //得到所有可能的右子树
            List<TreeNode> rightTrees = getRes(i + 1, end);
            //左右字数组合
            for (TreeNode leftTree : leftTrees) {
                for (TreeNode rightTree : rightTrees) {
                    //根节点
                    TreeNode root = new TreeNode(i);
                    root.left = leftTree;
                    root.right = rightTree;
                    //加入结果集
                    res.add(root);
                }
            }
        }
        return res;
    }

    /*
     * 96. 不同的二叉搜索树
     * 给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
     * */
    /*
    * 标签：动态规划
假设n个节点存在二叉排序树的个数是G(n)，令f(i)为以i为根的二叉搜索树的个数，则
    1、G(n) = f(1) + f(2) + f(3) + f(4) + ... + f(n)
        当i为根节点时，其左子树节点个数为i-1个，右子树节点为n-i，则
    2、f(i) = G(i-1)*G(n-i)
        综合两个公式可以得到 卡特兰数 公式
    3、G(n) = G(0)*G(n-1)+G(1)*(n-2)+...+G(n-1)*G(0)
    * */
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < n + 1; i++) {
            for (int j = 1; j < i + 1; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    /*98. 验证二叉搜索树
    * 给定一个二叉树，判断其是否是一个有效的二叉搜索树。
    假设一个二叉搜索树具有如下特征：
        节点的左子树只包含小于当前节点的数。
        节点的右子树只包含大于当前节点的数。
        所有左子树和右子树自身必须也是二叉搜索树。

    * */
    //二叉搜索树中序遍历为升序
    public boolean isValidBST(TreeNode root) {
        long pre = Long.MIN_VALUE;
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || root != null) {
            if (root != null) {
                stack.push(root);
                root = root.left;
            } else {
                root = stack.pop();
                // 如果中序遍历得到的节点的值小于等于前一个 inorder，说明不是二叉搜索树
                if (root.val <= pre) {
                    return false;
                }
                pre = root.val;
                root = root.right;
            }
        }
        return true;
    }

    /*
    100. 相同的树
    给定两个二叉树，编写一个函数来检验它们是否相同。
    如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
    * */
//    终止条件与返回值：
//    当两棵树的当前节点都为 null 时返回 true
//    当其中一个为 null 另一个不为 null 时返回 false
//    当两个都不为空但是值不相等时，返回 false
//    执行过程：当满足终止条件时进行返回，不满足时分别判断左子树和右子树是否相同，其中要注意代码中的短路效应
    /*
    * 针对树的框架：
    * void traverse(TreeNode root) {
    // root 需要做什么？在这做。
    // 其他的不用 root 操心，抛给框架
    traverse(root.left);
    traverse(root.right);
    }
    * 针对二叉搜索树
    * void BST(TreeNode root, int target) {
    if (root.val == target)
        // 找到目标，做点什么
    if (root.val < target)
        BST(root.right, target);
    if (root.val > target)
        BST(root.left, target);
    }
    * */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        //    当两棵树的当前节点都为 null 时返回 true
        if (p == null && q == null)
            return true;
        //    当其中一个为 null 另一个不为 null 时返回 false
        if (p == null || q == null)
            return false;
        //    当两个都不为空但是值不相等时，返回 false
        if (p.val != q.val)
            return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    /*
     * 101. 对称二叉树
     * 给定一个二叉树，检查它是否是镜像对称的。
     * */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetric(root.left, root.right);
    }

    private boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        return left.val == right.val && isSymmetric(left.right, right.left) && isSymmetric(left.left, right.right);
    }

    /*
     * 102. 二叉树的层序遍历
     * */
    List<List<Integer>> res_102 = new ArrayList<>();

    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return res_102;
        }
        Deque<TreeNode> deque = new ArrayDeque<>();
        TreeNode front = null;
        deque.add(root);
        while (!deque.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            //每一层的个数
            for (int i = deque.size(); i > 0; i--) {
                front = deque.poll();
                list.add(front.val);
                if (front.left != null) {
                    deque.add(front.left);
                }
                if (front.right != null) {
                    deque.add(front.right);
                }
            }
            res_102.add(list);
        }
        return res_102;
    }

    /*
    103. 二叉树的锯齿形层次遍历
    * */
    List<List<Integer>> res_103 = new ArrayList<>();

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) {
            return res_103;
        }
        Deque<TreeNode> deque = new ArrayDeque<>();
        TreeNode front = null;
        deque.add(root);
        int flag = 1;
        while (!deque.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            int[] temp = new int[deque.size()];
            for (int i = deque.size() - 1; i >= 0; i--) {
                front = deque.poll();
                temp[i] = front.val;
                if (front.left != null) {
                    deque.add(front.left);
                }
                if (front.right != null) {
                    deque.add(front.right);
                }
            }
            if (flag % 2 == 0) {
                for (int i = 0; i < temp.length; i++) {
                    list.add(temp[i]);
                }
            }
            if (flag % 2 != 0) {
                for (int i = temp.length - 1; i >= 0; i--) {
                    list.add(temp[i]);
                }
            }
            flag++;
            res_103.add(list);
        }
        return res_103;
    }

    /*
     * 104. 二叉树的最大深度
     * */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        //root深度比子树的最大深度+1
        return right > left ? right + 1 : left + 1;

    }

    /*
     * 105. 从前序与中序遍历序列构造二叉树
     * */
    int[] preOrd;
    HashMap<Integer, Integer> dic = new HashMap<>();

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        preOrd = preorder;
        for (int i = 0; i < inorder.length; i++) {
            dic.put(inorder[i], i);
        }
        return rebuild(0, 0, inorder.length - 1);
    }

    /**
     * @param pre_root 前序遍历中根节点的索引
     * @param in_left  中序遍历左边界
     * @param in_right 中序遍历右边界
     * @return
     */
    private TreeNode rebuild(int pre_root, int in_left, int in_right) {
        if (in_left > in_right) {
            return null;
        }
        //找出头节点
        TreeNode root = new TreeNode(preOrd[pre_root]);
        //找出头节点在中序中的索引位置
        int i = dic.get(preOrd[pre_root]);
        root.left = rebuild(pre_root + 1, in_left, i - 1);
        root.right = rebuild(pre_root + (i - in_left + 1), i + 1, in_right);
        return root;
    }

    /*
     * 106. 从中序与后序遍历序列构造二叉树
     * */
    int[] postOrd;
    HashMap<Integer, Integer> dic106 = new HashMap<>();

    public TreeNode buildTree106(int[] inorder, int[] postorder) {
        postOrd = postorder;
        for (int i = 0; i < inorder.length; i++) {
            dic106.put(inorder[i], i);
        }
        return rebuild109(postorder.length - 1, 0, inorder.length - 1);

    }

    private TreeNode rebuild109(int post_root, int in_left, int in_right) {
        if (in_left > in_right) {
            return null;
        }
        //头节点
        TreeNode root = new TreeNode(postOrd[post_root]);
        //找到在中序中的索引位置
        int i = dic106.get(postOrd[post_root]);
        root.right = rebuild109(post_root - 1, i + 1, in_right);
        root.left = rebuild109(post_root - (in_right - i) - 1, in_left, i - 1);
        return root;
    }

    /*
     * 107. 二叉树的层次遍历 II
     * 给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
     * */
    List<List<Integer>> res_107 = new ArrayList<>();

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if (root == null) {
            return res_107;
        }
        Deque<TreeNode> deque = new ArrayDeque<>();
        TreeNode front;
        deque.add(root);
        while (!deque.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                front = deque.poll();
                list.add(front.val);
                if (front.left != null) {
                    deque.add(front.left);
                }
                if (front.right != null) {
                    deque.add(front.right);
                }
            }
            res_107.add(0, list);
        }

//        swap(res_107);
        return res_107;
    }

    /*private void swap(List<List<Integer>> res_107) {
        int i = 0;
        int j = res_107.size() - 1;
        while (i< j) {
            List<Integer> list = new ArrayList<>(res_107.get(i));
            res_107.set(i, res_107.get(j));
            res_107.set(j, list);
            i++;
            j--;
        }
    }*/

    /*
     * 108. 将有序数组转换为二叉搜索树
     * */
    //根据根节点，就可以递归的生成左右子树。
//    注意这里的边界情况，包括左边界，不包括右边界。
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, start, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, end);
        return root;
    }

    /*
     * 109. 有序链表转换二叉搜索树
     * */
    public TreeNode sortedListToBST(ListNode head) {
        List<Integer> list = new ArrayList<>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        return sortedListToBST(list, 0, list.size() - 1);

    }

    private TreeNode sortedListToBST(List<Integer> list, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(list.get(mid));
        root.left = sortedListToBST(list, start, mid - 1);
        root.right = sortedListToBST(list, mid + 1, end);
        return root;
    }

    /*
     * 110. 平衡二叉树
     * 一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。
     * */
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        return Math.abs(treeDepth(root.left) - treeDepth(root.right)) <= 1
                && isBalanced(root.left)
                && isBalanced(root.right);

    }

    private int treeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = treeDepth(root.left);
        int right = treeDepth(root.right);
        return Math.max(left, right)+1;
    }
    /*
    * 111. 二叉树的最小深度
    * 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
    * */
    //叶子节点的定义是左孩子和右孩子都为 null 时叫做叶子节点
    //当 root 节点左右孩子都为空时，返回 1
    //当 root 节点左右孩子有一个为空时，返回不为空的孩子节点的深度
    //当 root 节点左右孩子都不为空时，返回左右孩子较小深度的节点值
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        //1.左孩子和有孩子都为空的情况，说明到达了叶子节点，直接返回1即可
        if (root.right == null && root.right == null) {
            return 1;
        }
        //2.如果左孩子和右孩子其中一个为空，那么需要返回比较大的那个孩子的深度
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if (root.left == null || root.right == null) {
            return left + right + 1;
        }
        //3.最后一种情况，也就是左右孩子都不为空，返回最小深度+1即可
        return Math.min(left, right) + 1;
    }

    /*
    * 112. 路径总和
    * 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
    * */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        // 如果是叶子结点，则看该结点值是否等于剩下的 sum
        if (root.left == null && root.right == null) {
            return sum == root.val;
        }
        // 每次遍历一个结点都要减去自己的值后重新递归
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    /*
     * 113. 路径总和 II
     * 给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。
     * */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        pathSum(res, root, sum, 0, new ArrayList<Integer>());
        return res;
    }

    private void pathSum(List<List<Integer>> res, TreeNode root, int sum, int cur, List<Integer> path) {
        cur += root.val;
        path.add(root.val);
        //找到符合的路径，就添加到res中，注意Java是引用类型，加入最终ans必须进行深拷贝
        if (cur == sum && root.left == null && root.right == null) {
            res.add(new ArrayList<>(path));
        }
        if (root.left != null) {
            pathSum(res, root.left, sum, cur, path);
            //递归结束，找到的结果都已添加
            path.remove(path.size() - 1);//移除最后的节点
        }
        if (root.right != null) {
            pathSum(res, root.right, sum, cur, path);
            //递归结束，找到的结果都已添加
            path.remove(path.size() - 1);//移除最后的节点
        }
    }

    /*
    * 114. 二叉树展开为链表
    * 给定一个二叉树，原地将它展开为一个单链表。
    * */
    //这就是一个递归的过程，递归的一个非常重要的点就是：不去管函数的内部细节是如何处理的，我们只看其函数作用以及输入与输出。对于函数flatten来说：
    //函数作用：将一个二叉树，原地将它展开为链表
    //输入：树的根节点
    //输出：无
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        //将根节点左子树变成链表
        flatten(root.left);
        //右子树
        flatten(root.right);
        TreeNode temp = root.right;
        //将树的右子树换成左子树
        root.right = root.left;
        //左子树置为空
        root.left = null;
        //找到右子树最后一个节点
        while (root.right != null) {
            root = root.right;
        }
        root.right = temp;
    }

    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }
        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    /*
    * 116. 填充每个节点的下一个右侧节点指针
    * */
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        if (root.left != null) {
            root.left.next = root.right;
        }
        if (root.right != null) {
            root.right.next = (root.next != null ? root.next.left : null);
        }
        connect(root.left);
        connect(root.right);
        return root;
    }
    /*
    * 117. 填充每个节点的下一个右侧节点指针 II
     * */
    public Node connect117(Node root) {
        //需要短路
        if (root == null||(root.left==null&&root.right==null)) {
            return root;
        }
        if (root.left != null && root.right != null) {
            root.left.next = root.right;
            //寻找下一个节点
            root.right.next = getNextNode(root);
        }
        if (root.left == null) {
            root.right.next=getNextNode(root);
        }
        if (root.right == null) {
            root.left.next = getNextNode(root);
        }
        //先递归生成右子树，因为左子树的next需要右子树
        connect(root.right);
        connect(root.left);
        return root;
    }

    private Node getNextNode(Node node) {
        //寻找头节点的next节点
        while (node.next != null) {
            if (node.next.left != null) {
                return node.next.left;
            }
            if (node.next.right != null) {
                return node.next.right;
            }
            node = node.next;
        }
        return null;
    }

    public static void main(String[] args) {
        int[] s1 = new int[]{0};
        int[] s2 = new int[]{1};
        ListNode ListNode1 = new ListNode(3);
        ListNode ListNode2 = new ListNode(5);
        ListNode1.next = ListNode2;
        new TestCode().reverseBetween(ListNode1, 1, 2);
        new TestCode().merge(s1, 0, s2, 1);


    }
}
