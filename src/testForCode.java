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
    public static ListNode deleteDuplication(ListNode head) {
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

    /*
    24：反转链表

    * */
    //头插法
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode pre = new ListNode(0);
        ListNode cur = head;
        ListNode nextNode;
        while (cur != null) {
            nextNode = cur.next;
            cur.next = pre.next;
            pre.next = cur;
            cur = nextNode;
        }
        return pre.next;
    }

    /*
     * 递归
     * 链表的后部分与head节点逆转，head的下一个节点的next指向head，同时将head.next = null，递归方法返回的是链表head的后部分
     * */
    public ListNode reverseList1(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode node = reverseList1(head.next);
        head.next.next = head;
        head.next = null;
        return node;
    }

    /*
    25. 合并两个排序的链表
    输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

    示例1：
    输入：1->2->4, 1->3->4
    输出：1->1->2->3->4->4

    * */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        ListNode pre = new ListNode(0);
        ListNode cur = pre;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if (l1 == null) {
            cur.next = l2;
        } else {
            cur.next = l1;
        }
        return pre.next;
    }

    //***************递归********************
    public ListNode mergeTwoLists1(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists1(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists1(l1, l2.next);
            return l2;
        }

    }

    /*
     * 35：复杂链表的复制
     * 题目要求：在复杂链表中，每个节点除了有一个next指针指向下一个节点，
     * 还有一个random指针指向链表中的任意节点或null，请完成一个能够复制复杂链表的函数。
     * */
    class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    //方法一：先遍历生成新链表，然后遍历连接随机节点
    //时间复杂度O(n^2) 空间O(1)
    public Node copyRandomList(Node head) {
        //遍历复制
        if (head == null) {
            return null;
        }
        Node cur = head.next;
        Node newHead = new Node(head.val);
        Node newPre = newHead;
        Node newCur = null;
        while (cur != null) {
            newCur = new Node(cur.val);
            newPre.next = newCur;
            cur = cur.next;
            newPre = newCur;
        }
        //指针回到头节点
        cur = head;
        newCur = newHead;
        Node temp;
        Node newTemp;
        while (cur != null) {
            if (cur.random != null) {
                temp = head;
                newTemp = newHead;
                while (temp != cur.random) {
                    temp = temp.next;
                    newTemp = newTemp.next;
                }
                newCur.random = newTemp;
            }
            cur = cur.next;
            newCur = newCur.next;
        }
        return newHead;

    }

    //方法二：使用哈希表来代替遍历寻找random节点
    public Node copyRandomList1(Node head) {
        //遍历复制
        if (head == null) {
            return null;
        }
        HashMap<Node, Node> res = new HashMap<>();
        Node newHead = new Node(head.val);
        res.put(head, newHead);
        Node cur = head.next;
        Node newPre = newHead;
        Node newCur = null;
        while (cur != null) {
            newCur = new Node(cur.val);
            res.put(cur, newCur);
            newPre.next = newCur;
            cur = cur.next;
            newPre = newCur;
        }
        //指针回到头节点
        cur = head;
        newCur = newHead;
        while (cur != null) {
            if (cur.random != null) {
                newCur.random = res.get(cur.random);
            }
            cur = cur.next;
            newCur = newCur.next;
        }
        return newHead;

    }

    public Node copyRandomList2(Node head) {
        if (head == null) {
            return null;
        }
        cloneNodes(head);
        connectRandomNodes(head);
        return reconnectNodes(head);
    }

    //a-b-c=>a-a'-b-b'-c-c'
    private void cloneNodes(Node head) {
        Node cur = head;
        Node temp = new Node(0);
        while (cur != null) {
            temp.val = cur.val;
            cur.next = temp;
            cur = temp;
        }
    }

    private void connectRandomNodes(Node head) {
        Node cur = head;
        Node curNext = head.next;
        while (true) {
            if (cur.random != null) {
                curNext.random = cur.random.next;
            }
            cur = cur.next.next;
            if (cur == null) {
                break;
            }
            curNext = curNext.next.next;
        }
    }

    private Node reconnectNodes(Node head) {
        Node cur = head;
        Node newHead = head.next;
        Node newCur = head.next;
        while (true) {
            cur.next = cur.next.next;
            cur = cur.next;
            if (cur == null) {
                break;
            }
            newCur.next = newCur.next.next;
            newCur = newCur.next;
        }
        return newHead;
    }

    /*
     * 52. 两个链表的第一个公共节点
     * 输入两个链表，找出它们的第一个公共节点。
     *
     * */
    //两个栈共同弹出，相同则是相同点
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        Stack<ListNode> a = new Stack<>();
        Stack<ListNode> b = new Stack<>();
        ListNode res = null;
        while (headA != null) {
            a.push(headA);
            headA = headA.next;
        }
        while (headB != null) {
            a.push(headB);
            headB = headB.next;
        }

        while (!a.empty() && !b.empty()) {
            if (a.peek().equals(b.peek())) {
                res = a.pop();
                b.pop();
            } else {
                break;
            }
        }
        return res;

    }


    //map存储
    public ListNode getIntersectionNode3(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        Map<ListNode, Integer> res = new HashMap<>();
        ListNode cur = headA;
        while (cur != null) {
            res.put(cur, cur.val);
            cur = cur.next;
        }
        cur = headB;
        while (cur != null) {
            if (res.containsKey(cur)) {
                return cur;
            }
            cur = cur.next;
        }
        return null;
    }

    //双指针相遇问题

    public ListNode getIntersectionNode1(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        ListNode a = headA;
        ListNode b = headB;
        while (a != b) {
            //指针a，b一起往后面走，走完的就从另一端重新开始，直至相遇
            a = (a == null) ? headB : a.next;
            b = (b == null) ? headA : b.next;
        }
        return a;
    }

    //********************栈队列*************************
    /*
    9：用两个栈实现队列
    思路：使用两个栈来实现，a实现存入功能
    出栈时把a栈的数据存入b栈再弹出
    插入肯定是往一个栈stack1中一直插入；删除时，直接出栈无法实现队列的先进先出规则，
    这时需要将元素从stack1出栈，压到另一个栈stack2中，然后再从stack2中出栈就OK了。
    需要稍微注意的是：当stack2中还有元素，stack1中的元素不能压进来；
    当stack2中没元素时，stack1中的所有元素都必须压入stack2中。
    * */
    class CQueue {

        public CQueue() {

        }

        Stack<Integer> a = new Stack<>();
        Stack<Integer> b = new Stack<>();


        public void appendTail(int value) {
            a.push(value);
        }

        public int deleteHead() {
            if (b.empty()) {
                if (a.empty()) {
                    return -1;
                }
                while (!a.empty()) {
                    b.push(a.pop());
                }
            }
            return b.pop();

        }
    }

    /*
    * 30：包含min函数的栈
    * 题目要求：
    定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的min函数。要求在该栈中，调用min，push及pop的时间复杂度都是o(1)。
    * */
    class MinStack {
        Stack<Integer> a;
        Stack<Integer> minStack;

        /**
         * initialize your data structure here.
         */
        public MinStack() {
            a = new Stack<>();
            minStack = new Stack<>();

        }

        public void push(int x) {
            a.push(x);
            if (minStack.empty() || x < minStack.peek()) {
                minStack.push(x);
            } else {
                minStack.push(minStack.peek());
            }
        }

        public void pop() {
            a.pop();
            minStack.pop();
        }

        public int top() {
            return a.peek();

        }

        public int min() {
            return minStack.peek();
        }
    }

    /*
    * 31：栈的压入弹出序列
    题目要求：
    输入两个整数序列，第一个序列表示栈的压入顺序，判断第二个序列是否为该栈的弹出序序列。
    假设压入栈的所有数字均不相等。例如，压入序列为(1,2,3,4,5)，序列(4,5,3,2,1)是它的弹出序列，而(4,3,5,1,2)不是。
    *
    * 思路：步骤1：栈压入序列第一个元素，弹出序列指针指弹出序列的第一个；
步骤2：判断栈顶元素是否等于弹出序列的第一个元素：
    步骤2.1：如果不是，压入另一个元素，进行结束判断，未结束则继续执行步骤2；
    步骤2.2：如果是，栈弹出一个元素，弹出序列指针向后移动一位，进行结束判断，未结束则继续执行步骤2；

结束条件：如果弹出序列指针还没到结尾但已经无元素可压入，则被测序列不是弹出序列。
         如果弹出序列指针以判断完最后一个元素，则被测序列是弹出序列。
    * */
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        if (popped == null || popped == null || popped.length != pushed.length) {
            return false;
        }
        int a = 0;
        int b = 0;
        Stack<Integer> res = new Stack<>();
        //当弹出数组到达末尾即成功
        while (b < popped.length) {
            if (res.isEmpty() || res.peek() != popped[b]) {
                //栈顶与弹出数组不相同则继续压入
                if (a < pushed.length) {
                    res.push(pushed[a++]);
                } else {
                    //弹出序列未结束但无元素可压入则失败
                    return false;
                }
            } else {
                //栈顶与弹出数组相同则弹出
                res.pop();
                b++;
            }
        }
        return true;
    }

    /*
    * 59：滑动窗口的最大值
题目要求：
给定一个数组和滑动窗口的大小，请找出所有滑动窗口的最大值。
* 例如，输入数组{2,3,4,2,6,2,5,1}和数字3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}。

    * */
    public int[] maxSlidingWindow(int[] nums, int k) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return new int[0];
        }

        for (int i = 0; i <= nums.length - k; i++) {
            int max = nums[i];
            for (int j = i + 1; j < i + k; j++) {
                max = nums[j] > max ? nums[j] : max;
            }
            res.add(max);
        }
        int size = res.size();
        int[] resArray = new int[size];
        for (int i = 0; i < size; i++) {
            resArray[i] = res.get(i);
        }
        return resArray;
    }

    //设置一个双向队列，先把k个元素按顺序存入
    //后面队列出列只有两个条件：1、超出边界把首个弹出。2、末项小于压入元素时弹出末项
    public int[] maxSlidingWindow1(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return new int[0];
        }
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < k - 1; i++) {
            while (!queue.isEmpty() && nums[i] > nums[queue.getLast()]) {
                queue.removeLast();
            }
            queue.addLast(i);
        }
        for (int i = k - 1; i < nums.length; i++) {
            //越界则弹出首项
            while (!queue.isEmpty() && i - queue.getFirst() + 1 > k) {
                queue.removeFirst();
            }
            //末项小于元素时弹出末项
            while (!queue.isEmpty() && nums[queue.getLast()] < nums[i]) {
                queue.removeLast();
            }
            //压入元素
            queue.addLast(i);
            res[i - (k - 1)] = nums[queue.getFirst()];
        }
        return res;
    }

    /*
    * 59.2：队列的最大值

    题目要求：
    定义一个队列并实现函数max得到队列里的最大值。要求max，pushBack，popFront的时间复杂度都是o(1)。
    * 思路：维持一个队列和一个双向递减队列
        用一个队列保存正常元素，另一个双向队列保存单调递减的元素
        入栈时，第一个队列正常入栈；第二个队列是递减队列，所以需要与之前的比较，从尾部把小于当前value的全部删除（因为用不到了）
        出栈时，第一个队列正常出栈；第二个队列的头部与出栈的值作比较，如果相同，那么一起出栈

    * */
    class MaxQueue {
        Queue<Integer> queue;
        Deque<Integer> deque;

        public MaxQueue() {
            queue = new ArrayDeque<>();
            deque = new ArrayDeque<>();

        }

        public int max_value() {
            if (deque.isEmpty()) {
                return -1;
            }
            return deque.getFirst();

        }

        public void push_back(int value) {
            queue.add(value);
            while (!deque.isEmpty() && deque.getLast() < value) {
                deque.removeLast();
            }
            deque.addLast(value);

        }

        public int pop_front() {
            if (queue.isEmpty()) {
                return -1;
            }
            int res = queue.poll();
            if (res == deque.peek()) {
                deque.removeFirst();
            }
            return res;

        }
    }

    //*********************树**************************
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    /*
    * 07. 重建二叉树
    * 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
        前序遍历 preorder = [3,9,20,15,7]
        中序遍历 inorder = [9,3,15,20,7]
         3
        / \
       9  20
         /  \
        15   7
    * */
    HashMap<Integer, Integer> dic = new HashMap<>();
    int[] preOr;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        preOr = preorder;
        //建立中序索引map方便取头节点index
        for (int i = 0; i < inorder.length; i++) {
            dic.put(inorder[i], i);
        }
        return rebuildTree(0, 0, inorder.length - 1);
    }

    /**
     * @param pre_root 前序遍历中根节点的索引
     * @param in_left  中序遍历左边界
     * @param in_right 中序遍历右边界
     * @return
     */
    private TreeNode rebuildTree(int pre_root, int in_left, int in_right) {
        if (in_left > in_right) {
            return null;
        }
        TreeNode root = new TreeNode(preOr[pre_root]);
        //找出头节点再后序遍历中的索引值
        int i = dic.get(preOr[pre_root]);
        //左子树： 根节点索引为 pre_root + 1 ，中序遍历的左右边界分别为 in_left 和 i - 1。
        root.left = rebuildTree(pre_root + 1, in_left, i - 1);
        // 根节点索引为 i - in_left + pre_root + 1（即：根节点索引 + 左子树长度 + 1），中序遍历的左右边界分别为 i + 1 和 in_right 。
        root.right = rebuildTree(pre_root + (i - in_left) + 1, i + 1, in_right);
        return root;
    }

    /*
    * 8.二叉树的下一个节点
    * 题目要求：
        给定二叉树和其中一个节点，找到中序遍历序列的下一个节点。树中的节点除了有左右孩子指针，还有一个指向父节点的指针。
        *         // 测试用例使用的树
        //            1
        //          // \\
        //         2     3
        //       // \\
        //      4     5
        //    inorder : 42513
    * */
    /*
    * 思路：
（1）如果输入的当前节点有右孩子，则它的下一个节点即为该右孩子为根节点的子树的最左边的节点，比如2->5,1->3
（2）如果输入的当前节点没有右孩子，就需要判断其与自身父节点的关系：
（2.1）如果当前节点没有父节点，那所求的下一个节点不存在，返回null.
（2.2）如果输入节点是他父节点的左孩子，那他的父节点就是所求的下一个节点,比如4->2
（2.3）如果输入节点是他父节点的右孩子，那就需要将输入节点的父节点作为新的当前节点，返回到（2）,
判断新的当前节点与他自身父节点的关系,比如5->1
    * */
    public class TreeLinkNode {
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode father = null;

        TreeLinkNode(int val) {
            this.val = val;
        }
    }

    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode.right != null) {
            pNode = pNode.right;
            while (pNode.left != null) {
                pNode = pNode.left;
            }
            return pNode;
        }
        while (pNode.father != null) {
            if (pNode.father.left == pNode) {
                return pNode.father;
            }
            pNode = pNode.father;
        }
        return null;
    }

    /*
    * 26. 树的子结构
    * 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
        B是A的子结构， 即 A中有出现和B相同的结构和节点值。
- 解题思路：
当A有一个节点与B的根节点值相同时，则需要从A的那个节点开始严格匹配，对应于下面的isMatch函数。
如果匹配不成功，则对它的左右子树继续判断是否与B的根节点值相同，重复上述过程。

    * */
    class Solution {
        public boolean isSubStructure(TreeNode a, TreeNode b) {
            if (b == null || a == null) {
                return false;
            }
            if (a.val == b.val) {
                //节点值相同时则进行逐个比较。否则的话还要去寻找a树中另一个相同的节点
                if (isMatch(a, b)) {
                    return true;
                }
            }
            return isSubStructure(a.left, b) || isSubStructure(a.right, b);

        }

        private boolean isMatch(TreeNode a, TreeNode b) {
            //子树为空则表示遍历结束，匹配
            if (b == null) {
                return true;
            }
            if (a == null) {
                return false;
            }
            if (a.val == b.val && isMatch(a.left, b.left) && isMatch(a.right, b.right)) {
                return true;
            }
            return false;
        }
    }

    /*
    * 26.二叉树的镜像
    * 题目要求：
        求一棵二叉树的镜像。

    - 解题思路：
        二叉树的镜像，即左右子树调换。从上到下，递归完成即可。
    * */
    public TreeNode mirrorTree(TreeNode root) {
        //递归函数的终止条件，节点为空时返回
        if (root == null) {
            return null;
        }
        if (root.left == null && root.right == null) {
            return root;
        }
        //交换
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        mirrorTree(root.left);
        mirrorTree(root.right);
        return root;
    }

    /*
    * 28.对称的二叉树
    * 题目要求：
        判断一棵二叉树是不是对称的。如果某二叉树与它的镜像一样，称它是对称的。
        * -思路：分析左右子树，左树的左子树等于右树的右子树，左树的右子树等于右树的左子树，
        * 对应位置刚好相反，判断两子树相反位置上的值是否相等即可
    * */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isEqual(root.left, root.right);
    }

    private boolean isEqual(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        //只有一边为空时不对称
        if (left == null || right == null) {
            return false;
        }
        return left.val == right.val && isEqual(left.left, right.right) && isEqual(left.right, right.left);
    }

    /*
    * 32：从上到下打印二叉树
    * 题目要求：
        从上到下打印二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
        - 层序遍历
    * */
    public int[] levelOrder(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        TreeNode front = null;
        if (root == null) {
            return new int[0];
        }
        queue.add(root);
        while (!queue.isEmpty()) {
            front = queue.poll();
            res.add(front.val);
            if (front.left != null) {
                queue.add(front.left);
            }
            if (front.right != null) {
                queue.add(front.right);
            }
        }
        //list->int[]
        int size = res.size();
        int[] resArray = new int[size];
        for (int i = 0; i < size; i++) {
            resArray[i] = res.get(i);
        }
        return resArray;
    }

    /*
     * 32 - II. 从上到下打印二叉树 II
     * 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
     * */
    //    3
    //   / \
    //  9  20
    //    /  \
    //   15   7
    //返回其层次遍历结果：
    //
    //[
    //  [3],
    //  [9,20],
    //  [15,7]
    //]
    //- 思路：同样使用队列，但要增加一个记录队列种剩余元素的个数：
    // 当遍历完一层时，队列种剩余的个数就是下一层队列种元素个数queue.size()
    public List<List<Integer>> levelOrderList(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new ArrayDeque<>();
        TreeNode front = null;
        queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            //注意初始化应该为队列中的个数
            for (int i = queue.size(); i > 0; i--) {
                front = queue.poll();
                temp.add(front.val);
                if (front.left != null) {
                    queue.add(front.left);
                }
                if (front.right != null) {
                    queue.add(front.right);
                }
            }
            res.add(temp);

        }
        return res;
    }

    /*
     * 32 - III. 从上到下打印二叉树 III
     * 请实现一个函数按照之字形顺序打印二叉树，
     * 即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
     * - 思路：每次清空队列就把下一层所有元素进队列，并且记录在一个数组种，之后通过flag判断行数，进行正向反向遍历。
     * */
    public List<List<Integer>> levelOrder1(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new ArrayDeque<>();
        int flag = 1;
        TreeNode front = null;
        queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            int[] nodeNum = new int[queue.size()];//记录每一层元素的数组
            for (int i = queue.size() - 1; i >= 0; i--) {//把当前层元素出列，并且把下一层元素进队列
                front = queue.poll();
                nodeNum[i] = front.val;//记录当前层的元素
                if (front.left != null) {
                    queue.add(front.left);
                }
                if (front.right != null) {
                    queue.add(front.right);
                }
            }
            if (flag % 2 == 0) {
                for (int j = 0; j < nodeNum.length; j++) {
                    temp.add(nodeNum[j]);
                }
            } else if (flag % 2 != 0) {
                for (int j = nodeNum.length - 1; j >= 0; j--) {
                    temp.add(nodeNum[j]);
                }
            }
            flag++;
            res.add(temp);
        }
        return res;
    }

    //- 官方方法：采用两个栈，对于不同层的结点，一个栈用于正向存储，一个栈用于逆向存储，打印出来就正好是相反方向。
    public List<List<Integer>> levelOrder2(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Stack<TreeNode> stack1 = new Stack<>();
        Stack<TreeNode> stack2 = new Stack<>();
        TreeNode front = null;
        stack1.push(root);

        while (!stack1.isEmpty() || !stack2.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            if (!stack1.isEmpty()) {
                while (!stack1.isEmpty()) {
                    front = stack1.pop();
                    temp.add(front.val);
                    if (front.left != null) {
                        stack2.push(front.left);
                    }
                    if (front.right != null) {
                        stack2.push(front.right);
                    }
                }
                //使用此方法才可以把temp中的元素复制进去
                res.add(new ArrayList<>(temp));
            } else {
                while (!stack2.isEmpty()) {
                    front = stack2.pop();
                    temp.add(front.val);
                    if (front.right != null) {
                        stack1.push(front.right);
                    }
                    if (front.left != null) {
                        stack1.push(front.left);
                    }
                }
                res.add(new ArrayList<>(temp));
            }
        }
        return res;
    }

    /*
    * 二叉排序树（搜索树）或者是一棵空树，或者是具有下列性质的二叉树：

　　　　（1）若左子树不空，则左子树上所有结点的值均小于或等于它的根结点的值；

　　　　（2）若右子树不空，则右子树上所有结点的值均大于或等于它的根结点的值；

　　　　（3）左、右子树也分别为二叉排序树；
    * 33：二叉搜索树的后序遍历
    * 题目要求：
        输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果，假设输入数组的任意两个数都互不相同
        *- 思路：二叉树后序遍历数组的最后一个数为根结点，剩余数字中，小于根结点的数字（即左子树部分）都排在前面，
        * 大于根结点的数字（即右子树部分）都排在后面。
    * */

    public boolean verifyPostorder(int[] postorder) {
        if (postorder.length == 0) {
            return true;
        }


        return isPost(postorder, 0, postorder.length - 1);


    }

    /**
     * @param postorder 数组
     * @param left      子树开始边界
     * @param right     子树右边界  有边界的元素作为根节点来区分下一层的左右子树
     * @return
     */
    private boolean isPost(int[] postorder, int left, int right) {
        if (left > right) {
            return true;
        }
        //判断左子树
        while (postorder[left] < postorder[right]) {
            left++;
        }
        int mid = left;
        for (int i = mid; i < right; i++) {
            if (postorder[i] < postorder[right]) {
                return false;
            }
        }

        return isPost(postorder, left, mid - 1) &&
                isPost(postorder, mid, right - 1);
    }

    /*
     * 34. 二叉树中和为某一值的路径
     * 输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。
     * 从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
     * */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        if (res == null) {
            return res;
        }
        pathSum(res, root, new ArrayList<Integer>(), sum, 0);
        return res;
    }

    //深度优先搜索
    //
    //path.remove(path.size() - 1)是将入栈的，已经遍历的节点取出
    //
    //注意Java是引用类型，加入最终ans必须进行深拷贝

    /**
     * @param res  结果集
     * @param root 遍历到的节点
     * @param path 记录的路径
     * @param sum  目标和
     * @param cur  目前记录元素的和
     */
    private void pathSum(List<List<Integer>> res, TreeNode root, List<Integer> path, int sum, int cur) {
        cur += root.val;
        path.add(root.val);
        //得到一个符合要求的路径时，创建一个新的ArrayList，拷贝当前路径到其中，并添加到lists中
        if (cur == sum && root.left == null && root.right == null) {
            res.add(new ArrayList<>(path));
        }
        if (root.left != null) {
            pathSum(res, root.left, path, sum, cur);
            //递归结束时，该留的路径已经记录了，不符合的路径也都不用理，删掉当前路径节点的值
            path.remove(path.size() - 1);
        }
        if (root.right != null) {
            pathSum(res, root.right, path, sum, cur);
            //递归结束时，该留的路径已经记录了，不符合的路径也都不用理，删掉当前路径节点的值
            path.remove(path.size() - 1);
        }
    }

    /*
     * 36. 二叉搜索树与双向链表
     * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。
     * - 思路：首先想一下中序遍历的大概代码结构（先处理左子树，再处理根结点，之后处理右子树），
     * 假设左子树处理完了，就要处理根结点，而根结点必须知道左子树的最大结点，
     * 所以要用函数返回值记录下来；之后处理右子树，右子树的最小结点（也用中序遍历得到）要和根结点链接。
     * */
    //- 思路：// 中序遍历，访问该节点的时候，对其做如下操作：
    //    // 1.将当前被访问节点curr的左孩子置为前驱pre（中序）
    //    // 2.若前驱pre不为空，则前驱的右孩子置为当前被访问节点curr
    //    // 3.将前驱pre指向当前节点curr，即访问完毕


    TreeNode pre = null; //全局变量pre

    public TreeNode treeToDoublyList(TreeNode root) {
        if (root == null) return root;
        TreeNode p = root;
        TreeNode q = root;
        while (p.left != null) {
            p = p.left;//最左节点
        }
        while (q.right != null) {
            q = q.right;//最右节点
        }
        inorder(root);
        // 上述形成的是一个非循环的双向链表
        // 需进行头尾相接
        p.left = q;
        q.right = p;

        return p;

    }

    private void inorder(TreeNode cur) {
        if (cur == null) {
            return;
        }
        inorder(cur.left);
        //遍历至此，cur为每个左子树的根节点
        cur.left = pre;
        if (pre != null) {
            pre.right = cur;
        }
        //cur退回根节点
        pre = cur;
        inorder(cur.right);

    }

    /*
    37. 序列化二叉树
    请实现两个函数，分别用来序列化和反序列化二叉树。
    -解释：把对象转换为字节序列的过程称为对象的序列化。
           把字节序列恢复为对象的过程称为对象的反序列化。

      对象的序列化主要有两种用途：
        1） 把对象的字节序列永久地保存到硬盘上，通常存放在一个文件中；
        2） 在网络上传送对象的字节序列

    - 注意：LinkedList<>可作为双端队列使用，null元素被允许
    dequeue null元素被禁止
    * */

    public static class Codec {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) {
                return "[]";
            }
            String res = "[";
            Queue<TreeNode> queue = new LinkedList<>();
            queue.add(root);
            while (!queue.isEmpty()) {
                TreeNode cur = queue.poll();
                if (cur != null) {
                    res += cur.val + ",";
                    queue.add(cur.left);
                    queue.add(cur.right);
                } else {
                    //空节点则添加null
                    res += "null,";
                }
            }
            //去除最后一个，
            res = res.substring(0, res.length() - 1) + "]";
            return res;

        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data == null || "[]".equals(data)) {
                return null;
            }
            //去除两边的边框
            String res = data.substring(1, data.length() - 1);
            String[] values = res.split(",");
            int index = 0;
            //头节点
            TreeNode head = generateTreeNode(values[index++]);
            Queue<TreeNode> queue = new LinkedList<>();
            TreeNode cur = null;
            queue.add(head);
            while (!queue.isEmpty()) {
                cur = queue.poll();
                cur.left = generateTreeNode(values[index++]);
                cur.right = generateTreeNode(values[index++]);

                if (cur.left != null) {
                    queue.add(cur.left);
                }
                if (cur.right != null) {
                    queue.add(cur.right);
                }
            }
            return head;
        }

        private TreeNode generateTreeNode(String value) {
            if ("null".equals(value)) {
                return null;
            }
            return new TreeNode(Integer.valueOf(value));

        }
    }
    /*
    * 54. 二叉搜索树的第k大节点
    * - 解题思路：
        二叉搜索树的中序遍历是有序的。可以引入一个计数器，每访问一个节点，计数器+1，当计数器等于k时，被访问节点就是该二叉搜索树的第k大节点。
        * - 注意：逆中序才是递减
    * */

    public int kthLargest(TreeNode root, int k) {
        //保证栈顶元素为cur的父节点
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        int i = 0;
        while (cur != null || !stack.isEmpty()) {
            if (cur != null) {
                stack.push(cur);
                cur = cur.right;
            } else {
                i++;
                if (i == k) {
                    return stack.peek().val;
                }
                cur = stack.pop().left;
            }
        }
        return 0;
    }

    /*
    * 55：二叉树的深度
    * -思路：解题思路：
        二叉树root的深度比其子树root.left与root.right的深度的最大值大1。因此可以通过上述结论递归求解。
        如果不使用递归，可以通过层序遍历（广度优先遍历）解决。
    * */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode cur = root;
        queue.add(cur);
        int count = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                cur = queue.poll();
                if (cur.left != null) {
                    queue.add(cur.left);
                }
                if (cur.right != null) {
                    queue.add(cur.right);
                }
            }
            count++;
        }
        return count;
    }

    //递归方法。root深度比子树的最大深度+1
    public int maxDepth1(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = maxDepth1(root.left);
        int right = maxDepth1(root.right);
        return left > right ? left + 1 : right + 1;

    }

    /*
     * 55 - II. 平衡二叉树
     * 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。
     * 如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
     * - 思路：计算树的深度，树的深度=max(左子树深度，右子树深度)+1。在遍历过程中，
     * 判断左右子树深度相差是否超过1，如果不平衡，则令树的深度=-1，用来表示树不平衡。
     * 最终根据树的深度是否等于-1来确定是否为平衡树。
     * */
    boolean isBalanced = true;

    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        treeDepth(root);
        return isBalanced;

    }

    private int treeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = treeDepth(root.left);
        int right = treeDepth(root.right);
        if (left - right > 1 || right - left > 1) {
            isBalanced = false;
        }
        return Math.max(left, right) + 1;
    }

    /*
    * 68 - I. 二叉搜索树的最近公共祖先
    * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
    百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，
    满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
- 说明有以下几种情况：
    二叉树本身为空，root == null ，return root
    p.val == q.val ,一个节点也可以是它自己的祖先
    p.val 和 q.val 都小于 root.val
    (两个子节点的值都小于根节点的值，说明它们的公共节点只能在二叉树的左子树寻找）
    p.val 和 q.val 都大于 root.val
    (两个子节点的值都大于根节点的值，说明它们的公共节点只能在二叉树的右子树寻找）
    如果上述的情况皆不满足，说明其公共节点既不在左子树也不在右子树上，只能为最顶端的公共节点，return root

    * */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        while (root != null) {
            if (p.val > root.val && q.val > root.val) {
                root = root.right;
            }
            if (p.val < root.val && q.val < root.val) {
                root = root.left;
            } else {
                break;
            }
        }
        return root;
    }

    //****************哈希****************************
    /*
     * 03. 数组中重复的数字
     * - 思路：建立hash表
     * */
    public int findRepeatNumber(int[] nums) {
        int[] hashTable = new int[nums.length];
        for (int num : nums) {
            if (hashTable[num] >= 1) {
                return num;
            }
            //没遇到过就相应位置置一
            hashTable[num] = 1;
        }
        return 0;
    }

    /*
     * 50. 第一个只出现一次的字符
     * 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格
     * -思路：字符（char）是长度为8的数据类型，共有256中可能，因此哈希表可以用一个长度为256的数组来代替
     * */
    public char firstUniqChar(String s) {
        char[] dic = new char[256];
        //第一遍遍历添加值
        for (int i = 0; i < s.length(); i++) {
            dic[s.charAt(i)]++;
        }
        //第二次寻找第一个出现一次的字符
        for (int i = 0; i < s.length(); i++) {
            if (dic[s.charAt(i)] == 1) {
                return s.charAt(i);
            }
        }
        return ' ';
    }

    //****************位运算**********************
    //左移，后空缺自动补0；
    //右移，分为逻辑右移和算数右移
    //1）逻辑右移 不管是什么类型，空缺自动补0；
    //2）算数右移 若是无符号数，则空缺补0，若是负数，空缺补1；
    /*
    * 15. 二进制中1的个数
    * 请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。
    * 例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2
    * -思路：根据 与运算 定义，设二进制数字 nn ，则有：
        若 n&1=0 ，则 n 二进制 最右一位 为 0 ；因为1除了最后一位各位都为0
        若 n&1=1 ，则 n 二进制 最右一位 为 1 。
- Java中无符号右移>>>
- 把一个整数减去1之后再和原来的整数做位与运算，得到的结果相当于把原整数的二进制表示形式的最右边的1变成0
    * */

    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int res = 0;
        while (n != 0) {
            res += n & 1;
            n >>>= 1;
        }
        return res;

    }
    //求负数的二进制的步骤：给定一个数，比如 12，我们能求得它的二进制 1100，如何求 −12 的二进制？
    // 实际上二进制前面有个符号位，正数前面符号位是 0，负数前面符号位是 1，12 的二进制实际上是 01100，
    // 那么求 −12 的二进制有两步：
    //
    //首先把符号位从 0 改成 1，然后对 12 每位取反。变成 10011
    //最后 +1，即 10011+1 = 10100，这就是 −12 的二进制

    /*
56：数组中只出现一次的两个数字
    题目要求：
    一个整数数组里除了两个数字出现一次，其他数字都出现两次。请找出这两个数字。要求时间复杂度为o(n)，空间复杂度为o(1)。
- 异或：位数上不相同为1
-思路：1、全体异或以后得到的得到的数temp，因为有两个数不相同则异或肯定不为0
        2、求得二进制位最右边一位为1的数字
        3、用该位是否等于1来划分为两组，再求异或即为所求

- n&-n是求一个二进制数的最低位的1对应的数

        设x=8
        8的二进制位：0 0 0 0 1 0 0 0
        对8取反：1 1 1 1 0 1 1 1
        取反后加1: 1 1 1 1 1 0 0 0

        +8:0 0 0 0 1 0 0 0
        -8:1 1 1 1 1 0 0 0
        &: 0 0 0 0 1 0 0 0

        lowbit = 8 & (-8) = 8

    * */
    public int[] singleNumber(int[] nums) {
        int temp = nums[0];
        for (int i = 1; i < nums.length; i++) {
            temp ^= nums[i];
        }
        int[] res = new int[2];
        int lowbit = temp & -temp;
        for (int i = 0; i < nums.length; i++) {
            if ((nums[i] & lowbit) == lowbit) {//与lowbit与不变说明改位为0
                res[0] ^= nums[i];
            } else {
                res[1] ^= nums[i];
            }
        }
        return res;

    }

    /*
     * 56 - II. 数组中数字出现的次数 II
     * 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。
     * -思路：将所有数字的二进制表示的对应位都加起来，如果某一位能被三整除，那么只出现一次的数字在该位为0；反之，为1。
     * */
    public int singleNumber2(int[] nums) {
        int[] bitSum = new int[32];
        for (int i = 0; i < 32; i++) {
            bitSum[i] = 0;
        }
        for (int i = 0; i < nums.length; i++) {
            int bitMask = 1;
            for (int j = 31; j >= 0; j--) {
                int bit = nums[i] & bitMask;//注意nums[i]&bitMask不一定等于1或者0，有可能等于00010000
                if (bit != 0) {
                    bitSum[j]++;
                }
                bitMask <<= 1;
            }
        }
        int result = 0;
        for (int i = 0; i < 32; i++) {
            result = result << 1;
            result += (bitSum[i] % 3);
        }
        return result;
    }
    /*
    - 收获
　　1.判断某个数x的第n位（如第3位）上是否为1，

　　　　1）通过 x&00000100 的结果是否为0 来判断。（不能根据是否等于1来判断）

　　　　2）通过（x>>3)&1 是否为0 来判断

　　2.通过number&bitMask的结果是否为0（不能用1判断），bitMask=1不断左移，可以将一个数的二进制存储到32位的数组中。

        int number=100;
        int bitMask=1;
        for(int j=31;j>=0;j--) {
            int bit=number&bitMask;  //注意arr[i]&bitMask不一定等于1或者0，有可能等于00010000
            if(bit!=0)
                bits[j]=1;
            bitMask=bitMask<<1;
        }
        　　3.通过以下代码实现二进制转化为数字（注意左移语句的位置）：

        int result=0;
        for(int i=0;i<32;i++) {
            result=result<<1;
            result+=bits[i];
            //result=result<<1;  //不能放在后面，否则最前面一位就没了
        }
    * */

    /*
    * 65. 不用加减乘除做加法
    * 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。
    * - 解题思路：不能用四则运算，那只能通过位运算了。
* 其实四则运算是针对十进制，位运算是针对二进制，都能用于运算。
*
    1.两数进行异或：  0011^0101=0110 这个数字其实是把原数中不需进位的二进制位进行了组合
    2.两数进行与：    0011&0101=0001 这个数字为1的位置表示需要进位，而进位动作是需要向前一位进位
    3.左移一位：      0001<<1=0010
    此时我们就完成0011 + 0101 = 0110 + 0010的转换
    如此转换下去，直到其中一个数字为0时，另一个数字就是原来的两个数字的和
    * */


    public int add(int a, int b) {
        int sum = a ^ b;
        int carry = (a & b) << 1;
        int temp;
        while (carry != 0) {
            temp = sum;
            sum = temp ^ carry;
            carry = (temp & carry) << 1;
        }
        return sum;

    }
    /*
    * 不使用新的变量完成交换两个原有变量的值
    *  //基于加减法
        int a = 3;
        int b = 5;
        a = a + b;
        b = a - b;
        a = a - b;

      //基于异或法
        a = 3;
        b = 5;
        a = a ^ b;
        b = a ^ b;
        a = a ^ b;
    * */

    //*******************查找***************************
    /*
     * 04. 二维数组中的查找
     * 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
     * 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     * -思路：从右上角开始搜寻
     * */
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return false;
        int r = 0;
        int c = matrix[0].length - 1;//右上角坐标
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
    * 11：旋转数组的最小数字
    题目要求：
    把一个数组最开始的若干个元素搬到末尾成为数组的旋转，
    * 如1,2,3,4,5=>3,4,5,1,2；0,1,1,1,1=>1,1,1,0,1；0,1,1,1,1=>1,0,1,1,1。求一个原本递增的数组旋转后的最小数字
    *
    * */
    public int minArray(int[] numbers) {
        int i = 0;
        while (i < numbers.length - 1) {
            if (numbers[i] <= numbers[i + 1]) {
                i++;
            } else
                break;
        }
        if (i == numbers.length - 1) {
            return numbers[0];
        }
        return numbers[i + 1];
    }

    // [3, 4, 5, 1, 2]
    // [1, 2, 3, 4, 5]
    // 不能使用左边数与中间数比较，这种做法不能有效地减治

    // [1, 2, 3, 4, 5]
    // [3, 4, 5, 1, 2]
    // [2, 3, 4, 5 ,1]

    public int minArray1(int[] numbers) {
        int len = numbers.length;
        if (len == 0) {
            return 0;
        }
        int left = 0;
        int right = len - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (numbers[mid] > numbers[right]) {
                // [3, 4, 5, 1, 2]，mid 以及 mid 的左边一定不是最小数字
                // 下一轮搜索区间是 [mid + 1, right]
                left = mid + 1;
            } else if (numbers[mid] == numbers[right]) {
                // 只能把 right 排除掉，下一轮搜索区间是 [left, right - 1]
                right = right - 1;
            } else {
                // 此时 numbers[mid] < numbers[right]
                // mid 的右边一定不是最小数字，mid 有可能是，下一轮搜索区间是 [left, mid]
                right = mid;
            }
        }

        // 最小数字一定在数组中，因此不用后处理
        return numbers[left];
    }

    /*
    * 53 - I. 在排序数组中查找数字 I
    * 统计一个数字在排序数组中出现的次数。
    * - 注意：二分查找中mid = left + (right - left) / 2;可以防止溢出
    * 解题思路：
        排序数组，定位某一个数值的位置，很容易想到二分查找。
        * 分成两部分：求第一个出现该值的位置start，求最后一个出现该值得位置end，end-start+1即为所求。
    * */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int first = findFirst(nums, target);
        if (first == -1) {
            return 0;
        }
        int last = findLast(nums, target);
        return last - first + 1;
    }

    private int findLast(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            //在待搜索区间只要有 2 个元素的时候，mid = (left + right) >>> 1 只能取到左边那个元素，
            // 如果此时边界设置是 left = mid ，区间分不开，因此要改变下取整的行为，在括号里加 1 变成上取整。
            int mid = left + (right - left + 1) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] == target) {
                //求右边的元素，所以左边肯定不是
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    private int findFirst(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] == target) {//因为是求最开始出现的元素，所以相同情况下右边肯定不是
                right = mid;
            } else {
                //这时候nums[mid]>right
                right = mid - 1;
            }
        }
        if (nums[left] == target) {
            return left;
        }
        return -1;//没找到
    }

    /*
    * 53 - II. 0～n-1中缺失的数字
    * 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。
    * 在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
- 收获
　　1.对于在排序数组中查找某些特定的数字，可以对二分法稍加改造，实现所需的功能。
    * */

    public int missingNumber(int[] nums) {
        int i = 0;
        for (; i < nums.length; i++) {
            if (nums[i] != i) {
                break;
            }
        }
        return i + 1;
    }

    //使用二分法查找所需元素效率最高
    //当中间数字等于其下标时，我们在后半部分查找；
//　　　　当中间数字不等于其下标时，
//　　　　1）如果中间数字的前一个数字也不等于其下标，则在前半部分查找；
//　　　　2）如果中间数字的前一个数字等于其下标，则说明中间数字的下标即为我们所要找的数字。
    public int getMissingNumber(int[] arr) {
        if (arr == null || arr.length <= 0)
            return -1;
        int low = 0;
        int high = arr.length - 1;
        while (low <= high) {
            int mid = (low + high) >> 1;
            if (arr[mid] != mid) {
                if (mid == 0 || arr[mid - 1] == mid - 1)
                    return mid;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    }

    /*
     * 21. 调整数组顺序使奇数位于偶数前面
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
     * */
    public int[] exchange(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        while (l < r) {
            while (l < r & nums[l] % 2 != 0) {
                l++;
            }
            while (l < r & nums[r] % 2 == 0) {
                r--;
            }
            if (l < r) {
                int temp = nums[l];
                nums[l] = nums[r];
                nums[r] = temp;
            }
        }
        return nums;
    }

    /*
    * 39. 数组中出现次数超过一半的数字
    * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
    你可以假设数组是非空的，并且给定的数组总是存在多数元素。
    * - 思路：或者排序中间的数即为所求
    * */
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 1; i++) {
            int count = 1;
            int value = nums[i];
            while (i < nums.length - 1 && nums[i] == nums[i + 1]) {
                count++;

            }
            if (count > nums.length / 2) {
                return value;
            }
            continue;
        }
        return 0;
    }

    //- 思路一：数字次数超过一半，则说明：排序之后数组中间的数字一定就是所求的数字。
    //
    //利用partition()函数获得某一随机数字，其余数字按大小排在该数字的左右。若该数字下标刚好为n/2，则该数字即为所求数字；若小于n/2，则在右边部分继续查找；反之，左边部分查找。
    public int majorityElement1(int[] array) {
        if (array == null || array.length <= 0)
            return 0;
        int l = 0;
        int r = array.length - 1;
        int index = partition(array, l, r);
        while (index != array.length >> 1) {
            if (index < array.length >> 1) {
                index = partition(array, index + 1, r);
            } else {
                index = partition(array, l, index - 1);
            }
        }
        //判断次数是否超过一半
        int num = array[index];
        int count = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] == num) {
                count++;
            }
        }
        if (count > array.length >> 1) {
            return num;
        }
        return 0;
    }

    private int partition(int[] array, int l, int r) {
        int temp = array[l];
        while (l < r) {
            while (l < r && temp <= array[r]) {
                r--;
            }
            if (l < r) {
                array[l++] = array[r];
            }
            while (l < r && temp >= array[l]) {
                l++;
            }
            if (l < r) {
                array[r--] = array[l];
            }
        }
        array[l] = temp;
        return l;
    }


    //- 思路二：数字次数超过一半，则说明：该数字出现的次数比其他数字之和还多
    //
    //　　遍历数组过程中保存两个值：一个是数组中某一数字，另一个是次数。
    // 遍历到下一个数字时，若与保存数字相同，则次数加1，反之减1。若次数=0，则保存下一个数字，次数重新设置为1。
    // 由于要找的数字出现的次数比其他数字之和还多，那么要找的数字肯定是最后一次把次数设置为1的数字。

    //采用阵地攻守的思想：
    //　　第一个数字作为第一个士兵，守阵地；count = 1；
    //　　遇到相同元素，count++;
    //　　遇到不相同元素，即为敌人，同归于尽,count--；当遇到count为0的情况，又以新的i值作为守阵地的士兵，继续下去，到最后还留在阵地上的士兵，有可能是主元素。
    //　　再加一次循环，记录这个士兵的个数看是否大于数组一般即可
    public int majorityElement2(int[] array) {
        if (array == null || array.length <= 0)
            return 0;
        int num = array[0];
        int count = 1;
        for (int i = 1; i < array.length; i++) {
            if (count == 0) {
                num = array[i];
                count++;
            } else if (array[i] == num)
                count++;
            else
                count--;
        }
        if (count > 0) {
            int times = 0;
            for (int i = 0; i < array.length; i++) {
                if (array[i] == num) {
                    times++;
                }
            }
            if (times * 2 > array.length) {
                return num;
            }
        }
        return 0;
    }


    /*
     * 40. 最小的k个数
     * 输入整数数组 arr ，找出其中最小的 k 个数。
     * 例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
     * */
    //使用最小堆
    public int[] getLeastNumbers(int[] arr, int k) {
        //构建大顶堆,从最小非叶子节点开始
        for (int i = arr.length / 2 - 1; i >= 0; i--) {
            adjustHeap(arr, i, arr.length);
        }
        for (int i = arr.length - 1; i >= 0; i--) {
            int temp = arr[i];
            arr[i] = arr[0];
            arr[0] = temp;
            //调换头节点在进行调整
            adjustHeap(arr, 0, i);
        }
        //挑选出最小的k个元素

        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = arr[i];
        }
        return res;
    }

    /**
     * @param arr    数组
     * @param i      为0时即为调整
     * @param length 要调整数组元素数目
     */
    private void adjustHeap(int[] arr, int i, int length) {
        int temp = arr[i];
        for (int k = 2 * i + 1; k < length; k = k * 2 + 1) {//从i结点的左子结点开始，也就是2i+1处开始
            if (k + 1 < length && arr[k + 1] > arr[k]) {
                k++;
            }
            if (arr[k] > temp) {//如果子节点小于父节点，将子节点值赋给父节点（不用进行交换），i作为下次需要比较调整的坐标
                arr[i] = arr[k];
                i = k;
            } else
                break;
        }
        arr[i] = temp;//将temp值放到最终的位置。比较完后i即为最后所需要待的位置
    }

    /*
     * 41. 数据流中的中位数
     * 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
     * 如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
     * */
    //- 思路
    //　　所谓数据流，就是不会一次性读入所有数据，只能一个一个读取，每一步都要求能计算中位数。
    //
    //　　将读入的数据分为两部分，一部分数字小，另一部分大。小的一部分采用大顶堆存放，大的一部分采用小顶堆存放。
    // 当总个数为偶数时，使两个堆的数目相同，则中位数=大顶堆的最大数字与小顶堆的最小数字的平均值；
    // 而总个数为奇数时，使小顶堆的个数比大顶堆多一，则中位数=小顶堆的最小数字。
    //
    //　　因此，插入的步骤如下：
    //
    //　　1.若已读取的个数为偶数（包括0）时，两个堆的数目已经相同，将新读取的数插入到小顶堆中，从而实现小顶堆的个数多一。
    // 但是，如果新读取的数字比大顶堆中最大的数字还小，就不能直接插入到小顶堆中了 ，此时必须将新数字插入到大顶堆中，
    // 而将大顶堆中的最大数字插入到小顶堆中，从而实现小顶堆的个数多一。
    //
    //　　2若已读取的个数为奇数时，小顶堆的个数多一，所以要将新读取数字插入到大顶堆中，此时方法与上面类似。

    class MedianFinder {
        private PriorityQueue<Integer> minHeap;
        private PriorityQueue<Integer> maxHeap;

        /**
         * initialize your data structure here.
         */
        public MedianFinder() {
            minHeap = new PriorityQueue<>();
            maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    return o2 - o1;//降序
                }
            });
        }

        public void addNum(int num) {
            if (((minHeap.size() + maxHeap.size()) & 1) == 0) {//偶数时候，下一个数字加入小顶堆
                if (!maxHeap.isEmpty() && num < maxHeap.peek()) {
                    //如果元素小于大顶堆的最大元素，则先加入大顶堆，然后再把最大元素提取出来加入小顶堆
                    maxHeap.add(num);
                    num = maxHeap.poll();
                }
                //优先加入小顶堆
                minHeap.add(num);
            } else {//奇数时，下一个数字放入大顶堆
                if (!minHeap.isEmpty() && num > minHeap.peek()) {
                    minHeap.add(num);
                    num = minHeap.poll();
                }
                maxHeap.add(num);
            }
        }

        public double findMedian() {
            double median;
            if (((minHeap.size() + maxHeap.size() & 1) == 0)) {
                median = (maxHeap.peek() + minHeap.peek()) / 2.0;
            } else {
                median = minHeap.peek();
            }
            return median;
        }
    }
    //- 收获
    //　　1.最大最小堆可以用PriorityQueue实现，PriorityQueue默认是一个小顶堆，通过传入自定义的Comparator函数可以实现大顶堆：
    /*
    PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(new Comparator<Integer>(){ //大顶堆
    @Override
    public int compare(Integer i1,Integer i2){
        return i2-i1; //降序排列
    }
});
    * */
    //- 注意：i1-i2 是升序
    //PriorityQueue的常用方法有：poll(),offer(Object),size(),peek()等。
    //
    //　　2.平均值应该定义为double，且（a+b）/2.0 。
    //
    //　　3.往最大堆中插入数据时间复杂度是O(logn)，获取最大数的时间复杂度是O(1)。
    //
    //　　4.这道题关键在于分成两个平均分配的部分，奇偶时分别插入到最大最小堆中，利用最大最小堆性质的插入方法要掌握。


    /*
     * 45. 把数组排成最小的数
     * 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个
     * */

    public String minNumber(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = 0; j < nums.length - 1 - i; j++) {
                if (bigger(nums[j], nums[j + 1])) {
                    int temp = nums[j];
                    nums[j] = nums[j + 1];
                    nums[j + 1] = temp;
                }
            }
        }
        StringBuilder builder = new StringBuilder();

        for (int num : nums) {
            builder.append(num + "");
        }
        return builder.toString();
    }

    //if（a>b) true
    private boolean bigger(int num1, int num2) {
        String temp1 = num1 + "" + num2;
        String temp2 = num2 + "" + num1;
        if (temp1.compareTo(temp2) > 0) {//大于0就是大于
            return true;
        } else
            return false;

    }

    public String minNumber1(int[] nums) {
        //使用内置函数排序
        String[] strNums = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strNums[i] = String.valueOf(nums[i]);
        }
        //排序
        Arrays.sort(strNums, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return (o1 + o2).compareTo(o2 + o1);//升序
            }
        });
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < strNums.length; i++) {
            builder.append(strNums[i]);
        }
        return builder.toString();
    }

    /*
     * 51. 数组中的逆序对
     * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
     * */
    public int reversePairs(int[] nums) {
        int count = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            int temp = nums[i];
            for (int j = i + 1; j < nums.length; j++) {
                if (temp > nums[j]) {
                    count++;
                }
            }
        }
        return count;
    }

    //借助归并算法，在排序的过程中完成统计
    public int reversePairs1(int[] nums) {
        int len = nums.length;
        if (len < 2) {
            return 0;
        }
        int[] temp = new int[len];
        return sort(nums, 0, len - 1, temp);
    }

    private int sort(int[] nums, int l, int r, int[] temp) {
        if (l == r) {
            return 0;
        }
        int mid = (l + r) / 2;
        int leftPairs = sort(nums, l, mid, temp);
        int rightPairs = sort(nums, mid + 1, r, temp);
        int reversePairs = leftPairs + rightPairs;
        //判断左边最大值是否小于右边最小值
        if (nums[mid] < nums[mid + 1]) {
            return reversePairs;
        }
        int mergeNums = merge(nums, l, mid, r, temp);
        return mergeNums + reversePairs;

    }

    private int merge(int[] nums, int l, int mid, int r, int[] temp) {
        int i = l;
        int j = mid + 1;
        int t = 0;
        int res = 0;
        while (i <= mid && j <= r) {
            if (nums[i] <= nums[j]) {
                temp[t++] = nums[i++];
            } else {//前值大于后值时需要统计
                res += (mid - i + 1);
                temp[t++] = nums[j++];
            }
        }
        while (i <= mid) {
            temp[t++] = nums[i++];
        }
        while (j <= r) {
            temp[t++] = nums[j++];
        }
        t = 0;
        while (l <= r) {
            //将temp中的元素全部拷贝到原数组中
            nums[l++] = temp[t++];
        }
        return res;
    }

    //***********************动态规划***********************************
    //解题思路：
    //本题有动态规划算法的几个明显特征：
    //（1）是求最优解问题，如最大值，最小值；
    //（2）该问题能够分解成若干个子问题，并且子问题之间有重叠的更小子问题。

    //通常按照如下4个步骤来设计一个动态规划算法：
    //　　1.求一个问题的最优解
    //　　2.整体问题的最优解依赖各子问题的最优解
    //　　3.小问题之间还有相互重叠的更小的子问题
    //　　4.为了避免小问题的重复求解，采用从上往下分析和从下往上求解的方法求解问题

    /*
     * 14- I. 剪绳子
     * 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），
     * 每段绳子的长度记为 k[0],k[1]...k[m] 。请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？
     * 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

     * */

    public int cuttingRope(int n) {
        //列举特殊长度项
        if (n == 2) {
            return 1;
        }
        if (n == 3) {
            return 2;
        }
        int[] dp = new int[n + 1];//数组多开一个方便
        //例外，本身长度大于乘积
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;

        for (int i = 4; i <= n; i++) {//在这里n也需要计算所以边界要大于n
            int max = 0;
            //算不同长度的最大值乘积，再比较最大值
            for (int j = 1; j <= i / 2; j++) {
                if (dp[j] * dp[i - j] > max) {
                    max = dp[j] * dp[i - j];
                }
            }
            dp[i] = max;
        }
        return dp[n];
    }


    /*
     * 42. 连续子数组的最大和
     * 输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
     * */
    //动态规划，定义dp[i]表示以data[i]为末尾元素的子数组和的最大值
    // 递归公式：dp[i] =  data[i]          i=0或dp[i-1]<=0
    //          dp[i-1]+data[i]           i!=0且dp[i-1]>0
    public int maxSubArray(int[] nums) {
        if (nums.length == 0 || nums == null) {
            return 0;
        }
        int[] dp = new int[nums.length + 1];
        dp[0] = nums[0];
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (dp[i - 1] <= 0) {
                dp[i] = nums[i];
            } else
                dp[i] = dp[i - 1] + nums[i];
            if (dp[i] > max) {
                max = dp[i];
            }
        }
        return max;
    }

    /*
     * 46. 把数字翻译成字符串
     * 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
     * 一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

     * */
//- 思路：dp[r]表示r个数字可以有几种翻译方式
    //这道题的状态转移方程为：
    //          dp[i−1]             num[i]和num[i−1]不能合成一个字符
    //dp[i] {
    //          dp[i-1]+dp[i-2]     num[i]和num[i−1]能合成一个字符


    public int translateNum(int num) {
        if (num < 0) {
            return 0;
        }
        String str = String.valueOf(num);
        int len = str.length();
        int[] dp = new int[len + 1];//一般状态数组多申请一位可以防止空串情况
        dp[0] = 1;//默认空串也算一种
        dp[1] = 1;
        for (int i = 1; i < len; i++) {
            if (str.charAt(i - 1) == '0' || str.substring(i - 1, i + 1).compareTo("25") > 0) {
                dp[i + 1] = dp[i];
            } else {
                dp[i + 1] = dp[i] + dp[i - 1];
            }
        }
        return dp[str.length()];
    }

    /*
     * 47. 礼物的最大价值
     * 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。
     * 你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。
     * 给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？
     * - 思路：先把第一行和第一列数组累加好，
     * dp方程：dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
     * */
    public int maxValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        int dp[][] = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int j = 1; j < m; j++) {
            dp[j][0] = dp[j - 1][0] + grid[j][0];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    /*
    * 48. 最长不含重复字符的子字符串
    * 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
    *
    * */
    public int lengthOfLongestSubstring(String s) {

        int len = s.length();
        int left = 0;
        int max = 0;
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < len; i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)));
            }
            map.put(s.charAt(i), i+1);//所以这里最好value计算为值的后一个index
            max = Math.max(max, i - left+1 );//比较的时候是i 和 left距离
        }
        return max;

    }

    public static void main(String[] args) {
        StringBuffer s = new StringBuffer("d ");
        char[] s1 = {'a'};
        char[] s2 = {'.', '*'};
        String s3 = "42";
        int[] a = new int[]{1, 3, -1, -3, 5, 3, 6, 7, 3};
        ListNode ListNode1 = new ListNode(1);
        ListNode ListNode2 = new ListNode(2);
        ListNode ListNode3 = new ListNode(3);
        ListNode ListNode4 = new ListNode(4);
        ListNode ListNode5 = new ListNode(5);
        ListNode1.next = ListNode2;
        ListNode2.next = ListNode3;
        ListNode3.next = ListNode4;
        ListNode4.next = ListNode5;
        TreeNode root = new TreeNode(1);
        root.right = new TreeNode(2);
        root.right.left = new TreeNode(3);
        Codec codec = new Codec();
        codec.deserialize(codec.serialize(root));
        System.out.println(new testForCode().reversePairs1(new int[]{1, 3, 2, 3, 1}));
        new testForCode().lengthOfLongestSubstring("abcabcbb");
        System.out.println(Arrays.toString(new testForCode().reversePrint2(ListNode1)));
        System.out.println(new testForCode().cuttingRope(10));
    }
}
