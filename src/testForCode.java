import java.util.*;

public class testForCode {
    /*字符串*/
    /*(4) 替换空格
    请实现一个函数，把字符串中的每个空格替换成"%20"。例如输入“We are happy.”，则输出“We%20are%20happy.”

    思路：先计算出需要的总长度，然后从后往前进行复制和替换。。。则每个字符只需要复制一次即可。时间效率为O(n)。
    * */
    public String replaceSpace(StringBuffer str) {
        if (str == null || str.length() == 0) {
            return null;
        }
        int len = str.length();
        int originalIndex = len - 1;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                len += 2;
            }
        }
        str.setLength(len);
        int curIndex = len - 1;
        for (; curIndex >= 0 && curIndex != originalIndex; ) {
            if (str.charAt(originalIndex) == ' ') {
                str.setCharAt(curIndex--, '0');
                str.setCharAt(curIndex--, '2');
                str.setCharAt(curIndex--, '%');
            } else {
                str.setCharAt(curIndex--, str.charAt(originalIndex));
            }
            originalIndex--;
        }
        return str.toString();
    }

    /*
    * 19. 正则表达式匹配
    请实现一个函数用来匹配包括 '.' 和 '*' 的正则表达式。模式中的字符 '.' 表示任意一个字符，
    而 '*' 表示它前面的字符可以出现任意次（包含 0 次）。
    在本题中，匹配是指字符串的所有字符匹配整个模式。
    例如，字符串 "aaa" 与模式 "a.a" 和 "ab*ac*a" 匹配，但是与 "aa.a" 和 "ab*a" 均不匹配。
    * */
    public boolean match(char[] str, char[] pattern) {
        return matchStr(str, 0, pattern, 0);
    }

    private boolean matchStr(char[] str, int i, char[] pattern, int j) {
        if (i == str.length && j == pattern.length) {
            return true;
        }
        //到末端模式串为空，j到长度而i未到位
        if (i < str.length && j == pattern.length) {
            return false;
        }
        //模式串下一位为*时进行匹配
        if (j + 1 < pattern.length && pattern[j + 1] == '*') {
            //字符串与模式串第二位相同
            if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) {
                //分别代表*匹配多位、*匹配完毕，下移两位
                return matchStr(str, i + 1, pattern, j) || matchStr(str, i, pattern, j + 2);
            }
            //字符串与模式串第二位不相同
            else {
                //直接跳过，j下移两位相当于*匹配零位
                return matchStr(str, i, pattern, j + 2);
            }

        }
        //模式串第二位不是*
        else {
            //而是直接对位相等
            if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) {
                //直接字符串和模式串都下移一位
                return matchStr(str, i + 1, pattern, j + 1);
            }
            return false;

        }

    }

    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        //这其实是属于双序列型动态规划，我们开数组要开 n+1 ，这样对于空串的处理十分方便。结果就是 dp[n][m]
        //dp[i][j] 代表 A 的前 i个和 B的前 j个能否匹配
        //字符串为空的情况是特殊情况
        dp[0][0] = true;
        //当字符串为空时，dp[0][]可以推
        for (int i = 1; i <= n; i++) {
            if (p.charAt(i - 1) == '*') {
                dp[0][i] = dp[0][i - 2];
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                //模式串的一般情况
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                    //模式串的后一位为*
                } else if (p.charAt(j - 1) == '*') {
                    //模式串前一个字符匹配则有三种可能
                    if (s.charAt(i - 1) == p.charAt(j - 2) || p.charAt(j - 2) == '.') {
                        dp[i][j] |= dp[i][j - 1]; // a* counts as single a
                        dp[i][j] |= dp[i - 1][j]; // a* counts as multiple a
                        dp[i][j] |= dp[i][j - 2]; // a* counts as empty
                    } else if (s.charAt(i - 1) != p.charAt(j - 2)) {
                        //不匹配只有一种方式
                        dp[i][j] |= dp[i][j - 2]; // a* counts only as empty
                    }
                }
            }
        }
        return dp[m][n];

    }

    /*
     *58 - I. 翻转单词顺序
     * 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。
     * 为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。
     * */
    public String reverseWords(String s) {
        String res = "";
        s = s.trim();
        int start = s.length() - 1, end = 0;
        while (start >= 0) {
            while (s.charAt(start) == ' ') {
                start--;
            }
            end = start;
            while (s.charAt(start) != ' ') {
                start--;
                if (start < 0) {
                    break;
                }
            }
            s += s.substring(start + 1, end + 1) + " ";
        }
        return res.trim();
    }

    //先反转整个句子再反转每个单词
    public String ReverseSentence(char[] chars) {
        if (chars == null || chars.length == 0) {
            return String.valueOf(chars);
        }
        //反转整个句子
        reverseChar(chars, 0, chars.length - 1);
        //逐个单词反转
        int start = 0;
        int end = 0;
        while (start < chars.length) {
            while (end < chars.length && chars[end] != ' ') {
                end++;
            }
            reverseChar(chars, start, end - 1);
            start = ++end;
        }
        return String.valueOf(chars);

    }

    private void reverseChar(char[] chars, int start, int end) {
        while (start < end) {
            char temp = chars[start];
            chars[start] = chars[end];
            chars[end] = temp;
            start++;
            end--;
        }
    }

    /*
     * 面试题58 - II. 左旋转字符串
     * 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。
     * 请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。
     * */
    public String reverseLeftWords(String s, int n) {
        return s.substring(n, s.length()) + s.substring(0, n);
    }

    /*
    * 67：把字符串转换成整数
    * 写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。
    首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
    当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，
    作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
    该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
    注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
    在任何情况下，若函数不能进行有效的转换时，请返回 0。

    说明：
    假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。
    如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。

    * */
    public int strToInt(String str) {
        str = str.trim();
        int res = 0;
        //正负标记位
        int sign = 1;
        if (str.length() == 0 || str == null) {
            return res;
        }
        int i = 0;
        //判断正负号
        if (i < str.length() && str.charAt(i) == '+' || str.charAt(i) == '-') {
            sign = str.charAt(i) == '-' ? -1 : 1;
            i++;
        }
        while (i < str.length()) {
            if (str.charAt(i) >= '0' && str.charAt(i) <= '9') {
                //判断与最大值比较，大于模10或者等于模10最后以为大于7
                //过界
                if (res > Integer.MAX_VALUE / 10 || res == Integer.MAX_VALUE / 10 && str.charAt(i) - '0' > 7) {
                    return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
                }
                res = res * 10 + str.charAt(i) - '0';
                i++;
            } else {
                return res * sign;
            }
        }
        return res * sign;

    }

    //*************************************************************
    //************************链表*********************************

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    /*
     * 6：从尾到头打印链表
     * 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
     * */

    //***********递归法***************
    public int[] reversePrint(ListNode head) {
        //递归法1->2->3 ==> 2->3->1  2->3 ==> 3->2
        if (head == null) {
            return new int[0];
        }
        List<Integer> res = new LinkedList<>();
        printListFromTailToHead(res, head);
        //list->int[]
        int size = res.size();
        int[] resArray = new int[size];
        for (int i = 0; i < size; i++) {
            resArray[i] = res.get(i);
        }
        return resArray;

    }

    private void printListFromTailToHead(List<Integer> res, ListNode node) {
        if (node == null) {
            return;
        }
        if (node != null) {
            printListFromTailToHead(res, node.next);
        }
        res.add(node.val);
    }
    //************又头插法逆序******************

    public int[] reversePrint1(ListNode head) {
        if (head == null) {
            return new int[0];
        }
        ListNode pre = new ListNode(0);
        while (head != null) {
            ListNode cur = head;
            ListNode nextNode = cur.next;
            cur.next = pre.next;
            pre.next = cur;
            head = nextNode;
        }
        List<Integer> res = new LinkedList<>();
        head = pre.next;
        while (head != null) {
            res.add(head.val);
            head = head.next;
        }
        //list->int[]
        int size = res.size();
        int[] resArray = new int[size];
        for (int i = 0; i < size; i++) {
            resArray[i] = res.get(i);
        }
        return resArray;

    }

    //****************栈*******************
    public int[] reversePrint2(ListNode head) {
        if (head == null) {
            return new int[0];
        }
        Stack<Integer> stack = new Stack<>();
        while (head != null) {
            stack.add(head.val);
            head = head.next;
        }
        int[] resArray = new int[stack.size()];
        int i = 0;
        while (!stack.isEmpty()) {
            resArray[i++] = stack.pop();
        }
        return resArray;
    }

    /*
    * 18. 删除链表的节点
    * 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
    返回删除后的链表的头节点。
    * */
    public ListNode deleteNode(ListNode head, int val) {
        if (head == null) {
            return head;
        }
        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode cur = pre;
        while (cur.next != null) {
            if (cur.next.val == val) {
                cur.next = cur.next.next;
            } else
                cur = cur.next;
        }
        return pre.next;
    }

    /*
    18题目二：删除排序链表中重复的节点

    比如[1,2,2,3,3,3],删除之后为[1];

    解题思路：
    由于是已经排序好的链表，需要确定重复区域的长度，删除后还需要将被删去的前与后连接，
    所以需要三个节点pre,cur,post，cur-post为重复区域，删除后将pre与post.next连接即可。
    此外，要注意被删结点区域处在链表头部的情况，因为需要修改head。

    * */
    public static ListNode deleteDuplication(ListNode head){
        if (head == null || head.next == null) {
            return head;
        }
        ListNode pre = null;
        ListNode cur = head;
        ListNode post = head.next;
        boolean needDel = false;
        while (post != null) {
            if (cur.val == post.val) {
                needDel = true;
                post = post.next;
            } else if (needDel && cur.val != post.val) {
                //头节点也要删除
                if (pre == null) {
                    head = post;
                } else {
                    pre.next = post;
                }
                cur = post;
                post = post.next;
                needDel = false;
            } else if (cur.val != post.val) {
                pre = cur;
                cur = post;
                post = post.next;
            }
        }
        //遍历完了
        if (needDel && pre != null) {
            pre.next = null;
        }
        //遍历完了，pre为空则表示全部为空
        else if (needDel && pre == null) {
            head = null;
        }
        return head;

    }
    /*
    22：链表中倒数第k个节点
    求链表中倒数第k个节点。链表的尾节点定义为倒数第1个节点。

    思路：使用两个距离为k的指针向右移动
    * */
    public ListNode getKthFromEnd(ListNode head, int k) {
        if (head == null) {
            return null;
        }
        ListNode start = head;
        ListNode end = head;
        for (int i = 0; i < k; i++) {
            end = end.next;
        }
        while (end != null) {
            start = start.next;
            end = end.next;
        }
        return start;
    }

    /*
    * 23：链表中环的入口节点
    题目要求：
    假设一个链表中包含环，请找出入口节点。若没有环则返回null。
    *
    *
    * 思路：
    使用双指针，一个指针 fast 每次移动两个节点，一个指针 slow 每次移动一个节点。因为存在环，
    所以两个指针必定相遇在环中的某个节点上。假设相遇点在下图的 z1 位置，
    此时 fast 移动的节点数为 x+2y+z，slow 为 x+y，由于 fast 速度比 slow 快一倍，因此 x+2y+z=2(x+y)，得到 x=z。
    在相遇点，slow 要到环的入口点还需要移动 z 个节点，如果让 fast 重新从头开始移动，并且速度变为每次移动一个节点，
    那么它到环入口点还需要移动 x 个节点。在上面已经推导出 x=z，因此 fast 和 slow 将在环入口点相遇。
    * */
    public ListNode entryNodeOfLoop(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            //相等的话说明相遇了，证明有环
            if (fast == slow) {
                break;
            }
        }
        // fast到了链表尾部,说明链表无环
        if (fast == null || fast.next == null) {
            return null;
        }
        //fast重新出发
        fast = head;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }



    public static void main(String[] args) {
        StringBuffer s = new StringBuffer("d ");
        char[] s1 = {'a'};
        char[] s2 = {'.', '*'};
        String s3 = "42";
        ListNode ListNode1 = new ListNode(1);
        ListNode ListNode2 = new ListNode(2);
        ListNode ListNode3 = new ListNode(3);
        ListNode ListNode4 = new ListNode(4);
        ListNode ListNode5 = new ListNode(5);
        ListNode1.next = ListNode2;
        ListNode2.next = ListNode3;
        ListNode3.next = ListNode4;
        ListNode4.next = ListNode5;

        System.out.println(Arrays.toString(new testForCode().reversePrint2(ListNode1)));
        System.out.println(new testForCode().deleteNode(ListNode1, 5));
    }
}
