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
    * 14- II. 剪绳子 II
    * - 思路：会越界，使用贪心算法，考验数学方法
    * 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），
    * 每段绳子的长度记为 k[0],k[1]...k[m] 。请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？
    * 例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
        答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1
    * */
    //算法流程：
    //当 n ≤3 时，按照贪心规则应直接保留原数字，但由于题目要求必须剪成 m>1段，因此必须剪出一段长度为 1的绳子，即返回 n - 1 。
    //当 n>3时，求 n除以 3的 整数部分 a 和 余数部分 b （即n=3a+b ），并分为以下三种情况（设求余操作符号为 "⊙" ）：
    //当 b = 0 时，直接返回 3^a ⊙ 10000000073
    //当 b = 1 时，要将一个 1 + 3 转换为 2+2 ，因此返回 (3^(a-1) * 4)⊙ 1000000007
    //当 b = 2 时，返回 (3^a * 2) ⊙ 1000000007

    public int cuttingRope1(int n) {

        if (n <= 3) return n - 1;
        long res = 1;
        while (n > 4) {
            res *= 3;
            res = res % 1000000007;//防止溢出
            n -= 3;
        }
        return (int) (res * n % 1000000007);

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
            map.put(s.charAt(i), i + 1);//所以这里最好value计算为值的后一个index
            max = Math.max(max, i - left + 1);//比较的时候是i 和 left距离
        }
        return max;

    }
    //*******************其他************************
    /*
     * 10- I. 斐波那契数列
     * */
    //写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：
    //
    //F(0) = 0,   F(1) = 1
    //F(N) = F(N - 1) + F(N - 2), 其中 N > 1.

    public int fib(int n) {
        if (n == 0) {
            return 0;
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
            if (dp[i] > 1000000007) {
                dp[i] = dp[i] % 1000000007;
            }
        }
        return dp[n];

    }

    /*
     * 10- II. 青蛙跳台阶问题
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
     * - 思路：dp[n]表示n级台阶的跳法
     * dp[n]   dp[n-1]+dp[n-2]
     * */
    public int numWays(int n) {
        if (n == 0) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
            if (dp[i] > 1000000007) {
                dp[i] = dp[i] % 1000000007;
            }
        }
        return dp[n];
    }

    //******************回溯法************************
    //- 思路：通常可以使用LinkedList来代替栈实现回溯，使用removeLast()方法
    //result = []
    //def backtrack(路径, 选择列表):
    //    if 满足结束条件:
    //        result.add(路径)
    //        return
    //
    //for 选择 in 选择列表:
    //    做选择
    //    backtrack(路径, 选择列表)
    //    撤销选择
    /*
     * 12. 矩阵中的路径
     * 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
     * 路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，
     * 那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。


     * */
    public boolean exist(char[][] board, String word) {
        int rows = board.length;
        int cols = board[0].length;
        if (rows == 0 || cols == 0 || word == null || board == null) {
            return false;
        }
        boolean[][] isVisited = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (hasPath(board, i, j, word, isVisited, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     *
     * @param board 矩阵
     * @param row   当前行数
     * @param col   当前列数
     * @param word  条件字符
     * @param isVisited 访问列表
     * @param i     当前路径长度
     * @return
     */
    private boolean hasPath(char[][] board, int row, int col, String word, boolean[][] isVisited, int i) {
        //约束条件，满足就跳出
        if (i == word.length()) {//遍历的个数和数据相同就可以跳出
            return true;
        }
        if (row < 0 || col < 0 || row >= board.length || col >= board[0].length) {
            return false;
        }
        //递归
        //如果未被访问，且匹配字符串，标记当前位置为已访问，分上下左右四个位置递归求解
        if (!isVisited[row][col] && board[row][col] == word.charAt(i)) {
            isVisited[row][col] = true;
            boolean hashPath =
                    hasPath(board, row, col + 1, word, isVisited, i + 1) ||/*左*/
                            hasPath(board, row, col - 1, word, isVisited, i + 1) ||/*右*/
                            hasPath(board, row - 1, col, word, isVisited, i + 1) ||/*上*/
                            hasPath(board, row + 1, col, word, isVisited, i + 1);/*下*/
            if (hashPath) {
                return true;
            } else {
                //路径失败，回溯
                isVisited[row][col] = false;
                return false;
            }
        } else
            //如果已经被访问或者值不相同则直接返回
            return false;
    }

    /*
     * 13. 机器人的运动范围
     * 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。
     * 一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），
     * 也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。
     * 但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？
     * */
    public int movingCount(int m, int n, int k) {
        if (m == 0 || n == 0 || k == 0) {
            return 0;
        }
        boolean[][] isVisited = new boolean[m][n];
        return moveCount(m, n, 0, 0, k, isVisited);

    }

    private int moveCount(int m, int n, int row, int col, int k, boolean[][] isVisited) {
        int count = 0;
        if (canGet(k, m, n, row, col, isVisited)) {
            isVisited[row][col] = true;
            count = 1 + moveCount(m, n, row + 1, col, k, isVisited) +
                    moveCount(m, n, row - 1, col, k, isVisited) +
                    moveCount(m, n, row, col + 1, k, isVisited) +
                    moveCount(m, n, row, col - 1, k, isVisited);
        }
        return count;
    }

    //判断是否格子是否可以进入
    private boolean canGet(int k, int m, int n, int row, int col, boolean[][] isVisited) {
        return row >= 0 && col >= 0 && row < m && col < n && !isVisited[row][col]
                && (getDigitSum(row) + getDigitSum(col)) <= k;
    }

    private int getDigitSum(int num) {
        int sum = 0;
        while (num > 0) {
            sum += num % 10;
            num /= 10;
        }
        return sum;
    }

    /*
     * 16. 数值的整数次方
     * 实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。
     * */
    //- 思路：1）0的负数次方不存在；2）0的0次方没有数学意义；3）要考虑exponent为负数的情况。
    // 所以可以对exponent进行分类讨论，在对base是否为0进行讨论。
//- 注意：n 可以取到 -2147483648（整型负数的最小值），因此，在编码的时候，需要将 n 转换成 long 类型。
    public double myPow(double x, int n) {
        double result = 1;
        long N = n;//转换成long类型防止取反数越界
        if (N < 0) {
            x = 1 / x;
            N *= -1;
        }
        while (N > 0) {
            if ((N & 1) == 1) {
                result *= x;
            }
            x *= x;//底数翻倍
            N >>>= 1;//数右移一位
        }
        return result;
    }

    /*
     *17. 打印从1到最大的n位数
     * 输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。
     * - 注意：大数问题
     * */
    // TODO: 2020/3/8 大数问题
    public int[] printNumbers(int n) {
        int count = 9;
        for (int i = 0; i < n; i++) {
            count = count * 10 + 9;
        }
        int[] ints = new int[count];
        for (int i = 0; i < count; i++) {
            ints[i] = i + 1;
        }
        return ints;

    }

    /*
     * 20. 表示数值的字符串
     * 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
     * 例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"0123"及"-1E-16"都表示数值，
     * 但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。

     * */
    // TODO: 2020/3/9
    public boolean isNumber(String s) {
        return false;
    }

    /*
     * 29. 顺时针打印矩阵
     * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

     * */
    //-思路一：使用游标移动
    public int[] spiralOrder(int[][] matrix) {
        if (matrix.length == 0) {
            return new int[0];
        }
        int R = matrix.length;
        int C = matrix[0].length;
        int[] res = new int[R * C];
        boolean[][] isVisited = new boolean[R][C];
        int[] dr = {0, 1, 0, -1};//游标移动参数
        int[] dc = {1, 0, -1, 0};
        int r = 0, c = 0, di = 0;
        for (int i = 0; i < R * C; i++) {
            res[i] = matrix[r][c];
            isVisited[r][c] = true;
            int cr = r + dr[di];
            int cc = c + dc[di];
            if (cr >= 0 && cc >= 0 && cr < R && cc < C && !isVisited[cr][cc]) {
                r = cr;
                c = cc;
            } else {
                di = (di + 1) % 4;
                r += dr[di];
                c += dc[di];
            }
        }
        return res;
    }

    //- 思路二：设定边界法
    public int[] spiralOrder1(int[][] matrix) {
        if (matrix.length == 0) {
            return new int[0];
        }
        int rows = matrix.length;
        int cols = matrix[0].length;
        List<Integer> res = new ArrayList<>();
        int r1 = 0, r2 = rows - 1;
        int c1 = 0, c2 = cols - 1;
        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; c++) {
                res.add(matrix[r1][c]);
            }
            for (int r = r1 + 1; r <= r2; r++) {
                res.add(matrix[r][c2]);
            }
            if (r1 < r2 && c1 < c2) {//注意当不是正规的矩阵时，在判断循环时要注意只有单行的情况，避免重复
                for (int c = c2 - 1; c >= c1; c--) {
                    res.add(matrix[r2][c]);
                }
                for (int r = r2 - 1; r >= r1 + 1; r--) {
                    res.add(matrix[r][c1]);
                }
            }
            r1++;
            r2--;
            c1++;
            c2--;
        }
        int size = res.size();
        int[] resArray = new int[size];
        for (int i = 0; i < size; i++) {
            resArray[i] = res.get(i);
        }
        return resArray;
    }

    /*
     *  * 回溯法
     *
     * 字符串的排列和数字的排列都属于回溯的经典问题
     *
     * 回溯算法框架：解决一个问题，实际上就是一个决策树的遍历过程：
     * 1. 路径：做出的选择
     * 2. 选择列表：当前可以做的选择
     * 3. 结束条件：到达决策树底层，无法再做选择的条件
     *
     * 伪代码：
     * result = []
     * def backtrack(路径，选择列表):
     *     if 满足结束条件：
     *         result.add(路径)
     *         return
     *     for 选择 in 选择列表:
     *         做选择
     *         backtrack(路径，选择列表)
     *         撤销选择
     *
     * 核心是for循环中的递归，在递归调用之前“做选择”，
     * 在递归调用之后“撤销选择”。

     * */
    /*
     * 38. 字符串的排列
     * 输入一个字符串，打印出该字符串中字符的所有排列。
     * */
    /*
     * 交换法 —— 回溯算法
     *
     * [a, [b, c]]
     * [b, [a, c]] [c, [b, a]]
     *
     * 如上，对字符串"abc"分割，每次固定一个字符为一部分，
     * 其他字符为另一部分，再将固定字符与其他字符进行交换，
     * 依次遍历每个字符，再进行回溯递归。
     * */
    //对于a,b,c（下标为0,1,2）
    //0与0交换,得a,b,c => 1与1交换,得a,b,c =>2与2交换,得a,b,c(存入)
    //                => 1与2交换，得a,c,b =>2与2交换,得a,c.b(存入)
    //0与1交换,得b,a,c => 1与1交换,得b,a,c =>2与2交换,得b,a,c(存入)
    //                => 1与2交换，得b,c,a =>2与2交换,得b,c,a(存入)
    //0与2交换,得c,b,a => 1与1交换,得c,b,a =>2与2交换,得c,b,a(存入)
    //                => 1与2交换，得c,a,b =>2与2交换,得c,a.b(存入)

    public String[] permutation(String s) {
        //去重
        HashSet<String> res = new HashSet<String>();
        if (s.isEmpty()) {
            return new String[0];
        }
        //字符数组排序
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        permutationCore(res, chars, 0);
        String[] strs = new String[res.size()];
        int i = 0;
        for (String x : res) {
            strs[i++] = x;
        }
        return strs;
    }

    private void permutationCore(HashSet<String> res, char[] chars, int index) {
        if (index == chars.length) {
            res.add(String.valueOf(chars));
        }
        for (int i = index; i < chars.length; i++) {
            //先固定某一个元素
            swap(chars, index, i);
            //再去递归后面的元素
            permutationCore(res, chars, index + 1);
            //回溯，需把之前换过的元素换回原来的位置
            swap(chars, index, i);
        }
    }

    private void swap(char[] chars, int i, int j) {
        char tmp = chars[i];
        chars[i] = chars[j];
        chars[j] = tmp;
    }

    public String[] permutation1(String s) {
        //去重
        HashSet<String> res = new HashSet<>();
        if (s.isEmpty()) {
            return new String[0];
        }
        char[] chars = s.toCharArray();
        boolean[] isVisited = new boolean[s.length()];
        Arrays.sort(chars);
        backtrack(res, chars, isVisited, new StringBuilder());
        String[] strs = new String[res.size()];
        int i = 0;
        for (String x : res) {
            strs[i++] = x;
        }
        return strs;
    }

    private void backtrack(HashSet<String> res, char[] chars, boolean[] isVisited, StringBuilder builder) {
        if (builder.length() == chars.length) {
            res.add(builder.toString());
            return;
        }
        for (int i = 0; i < chars.length; i++) {
            if (isVisited[i]) {
                continue;
            }
            isVisited[i] = true;
            builder.append(chars[i]);
            backtrack(res, chars, isVisited, builder);
            //移除最后一个元素完成回溯
            builder.deleteCharAt(builder.length() - 1);
            isVisited[i] = false;
        }
    }


    //收获
    //　　1.要对字符串进行修改，可以将字符串转化为字符数组进行修改，也可以考虑使用StringBuilder类。
    //
    //　　2.list.contains()方法可以直接判断是否有重复字符串；Collections.sort(list)可以将list中的字符串进行排序。
    //
    //　　3.字符串和字符数组间的转化：str.toCharArray()     String.valueOf(strArray)
    //
    //　　4.数组在递归过程中进行了交换后，最终要记得交换回来（代码最后几行）相当于回溯

    /*
    * 43. 1～n整数中1出现的次数
    * 输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

    * */
    //-思路：递归法
    public int countDigitOne(int n) {
        return dfs(n);

    }

    private int dfs(int n) {
        if (n <= 0) {
            return 0;
        }

        String numStr = String.valueOf(n);
        int high = numStr.charAt(0) - '0';//1
        int pow = (int) Math.pow(10, numStr.length() - 1);//1000
        int last = n - high * pow;//234

        if (high == 1) {
            // 最高位是1，如1234, 此时pow = 1000,那么结果由以下三部分构成：
            // (1) dfs(pow - 1)代表[0,999]中1的个数;
            // (2) dfs(last)代表234中1出现的个数;
            // (3) last+1代表固定高位1有多少种情况。（000-234）
            return dfs(pow - 1) + dfs(last) + last + 1;
        } else {
            // 最高位不为1，如2234，那么结果也分成以下三部分构成：
            // (1) pow代表固定高位1，有多少种情况;（1000-1999）
            // (2) high * dfs(pow - 1)代表999以内和1999以内低三位1出现的个数;（high）个[0,999]
            // (3) dfs(last)同上。（高位不变）
            return pow + high * dfs(pow - 1) + dfs(last);
        }
    }

    /*
    * -思路二：找规律
    * 对于整数n，我们将这个整数分为三部分：当前位数字cur，更高位数字high，更低位数字low，如：对于n=21034，当位数是十位时，cur=3，high=210，low=4。

　　我们从个位到最高位 依次计算每个位置出现1的次数：

　　1）当前位的数字等于0时，例如n=21034，在百位上的数字cur=0，百位上是1的情况有：00100~00199，01100~01199，……，20100~20199。一共有21*100种情况，即high*100;

　　2）当前位的数字等于1时，例如n=21034，在千位上的数字cur=1，千位上是1的情况有：01000~01999，11000~11999，21000~21034。一共有2*1000+（34+1）种情况，即high*1000+(low+1)。

　　3）当前位的数字大于1时，例如n=21034，在十位上的数字cur=3，十位上是1的情况有：00010~00019，……，21010~21019。一共有（210+1）*10种情况，即(high+1)*10。
    * */

    public int countDigitOne1(int n) {
        //求每个位的数字所用
        int index = 1;
        int count = 0;
        int high = n, cur = 0, low = 0;
        while (high > 0) {  //i代表位数
            high /= 10; //更高位数字
            cur = (n / index) % 10;  //当前位数字
            low = n % index;  //更低位数字
            if (cur == 0) {
                count += high * index;
            }
            if (cur == 1) {
                count += high * index + (low + 1);
            }
            if (cur > 1) {
                count += (high + 1) * index;
            }
            index *= 10;
        }
        return count;
    }

    /*
     * 44. 数字序列中某一位的数字
     * 数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。
     * */
    public int findNthDigit(int n) {
        return 0;

    }

    /*
     * 57. 和为s的两个数字
     * 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
     * */
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> dic = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (!dic.containsKey(nums[i])) {
                dic.put(target - nums[i], nums[i]);
            } else {

                return new int[]{dic.get(nums[i]), nums[i]};
            }
        }
        return new int[2];
    }

    //- 思路：双指针
    public int[] twoSum1(int[] nums, int target) {
        int[] result = new int[2];
        if (nums == null || nums.length < 2) {
            return result;
        }
        int curSum = nums[0] + nums[nums.length - 1];
        int left = 0;
        int right = nums.length - 1;
        while (curSum != target && left < right) {
            if (curSum < target) {
                left++;
            } else
                right--;
            curSum = nums[left] + nums[right];
        }
        if (curSum == target) {
            result[0] = nums[left];
            result[1] = nums[right];
        }
        return result;
    }

    /*
     * 57 - II. 和为s的连续正数序列
     * 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
     * */
    //- 思路：首项加末项*项数/2

    public int[][] findContinuousSequence(int target) {
        LinkedList<int[]> res = new LinkedList<>();
        for (int i = 1; i < target / 2; i++) {
            List<Integer> temp = new ArrayList<>();
            int sum = 0;
            int start = i;
            while (true) {
                sum += start;
                temp.add(start++);
                if (sum >= target) {
                    break;
                }
            }
            if (sum == target) {
                //list->int[]
                int size = temp.size();
                int[] resArray = new int[size];
                for (int j = 0; j < size; j++) {
                    resArray[j] = temp.get(j);
                }
                res.add(resArray);
            }
        }
        return res.toArray(new int[0][]);
    }

    //-思路：依旧使用两个指针small，big，值分别为1，2。如果从small加到big的和等于s，即找到了一组解，然后让big后移，继续求解。
    // 如果和小于s，big后移，如果和大于s，small前移。直到small大于s/2停止。
    public int[][] findContinuousSequence1(int target) {
        LinkedList<int[]> res = new LinkedList<>();
        int small = 1, big = 2, middle = target >> 1;
        int curSum = small + big;
        while (small <= middle) {
            if (curSum == target) {
                int[] temp = new int[big - small + 1];
                int k = 0;
                for (int i = small; i <= big; i++) {
                    temp[k++] = small;
                }
                res.add(temp);
                big++;
                curSum += big;
            } else if (curSum < target) {
                big++;
                curSum += big;
            } else {
                curSum -= small;
                small++;
            }
        }
        return res.toArray(new int[0][]);
    }


    /*
     *60. n个骰子的点数
     * 把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
     * */
    //- 思路：状态转移
    /*
    * n个骰子点数和为s的种类数只与n-1个骰子的和有关。因为一个骰子有六个点数，那么第n个骰子可能出现1到6的点数。
    * 所以第n个骰子点数为1的话，f(n,s)=f(n-1,s-1)，当第n个骰子点数为2的话，f(n,s)=f(n-1,s-2)，…，依次类推。
    * 在n-1个骰子的基础上，再增加一个骰子出现点数和为s的结果只有这6种情况！
    * 那么有：f(n,s)=f(n-1,s-1)+f(n-1,s-2)+f(n-1,s-3)+f(n-1,s-4)+f(n-1,s-5)+f(n-1,s-6)
上面就是状态转移方程，已知初始阶段的解为：当n=1时, f(1,1)=f(1,2)=f(1,3)=f(1,4)=f(1,5)=f(1,6)=1。

    * */
    public double[] twoSum(int n) {
        int[][] dp = new int[n + 1][6 * n + 1];
        double[] res = new double[5 * n + 1];//6n-n+1
        double all = Math.pow(6, n);
        //特殊情况赋值
        for (int i = 1; i <= 6; i++) {
            dp[1][i] = 1;
        }
        //n个骰子
        for (int i = 1; i <= n; i++) {
            //i个骰子值范围
            for (int j = i; j <= 6 * n; j++) {
                //骰子取值范围为1-6
                for (int k = 1; k <= 6; k++) {
                    dp[i][j] += j > k ? dp[i - 1][j - k] : 0;//只有当j的取值大于k时才满足条件可以加入
                }
            }
        }
        for (int i = n; i <= 6 * n; i++) {
            res[i - n] = dp[n][i] / all;
        }
        return res;
    }

    /*
     * 61. 扑克牌中的顺子
     * 从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。
     * 2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
     * */
    //- 思路：　　1）进行对5张牌进行排序；
    //　　2）找出0的个数；
    //　　3）算出相邻数字的空缺总数；
    //　　4）如果0的个数大于等于空缺总数，说明连续，反之不连续；
    //　　5）记得判断相邻数字是否相等，如果有出现相等，说明不是顺子。

    public boolean isStraight(int[] nums) {
        Arrays.sort(nums);
        int numZero = 0;
        int numGap = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                numZero++;
            }
        }
        int small = numZero;
        int big = numZero + 1;
        while (big < nums.length) {
            if (nums[small] == nums[big]) {
                return false;
            }
            numGap += nums[big++] - nums[small++] - 1;//计算间隔累计，同时两个指针后移
        }
        if (numZero >= numGap) {
            return true;
        }
        return false;
    }

    /*
    * 62. 圆圈中最后剩下的数字
    * 0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。
    例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。
    * */
    //- 思路：1.采用链表来存放数据，每次对长度取余来实现循环，LinkedList比ArrayList更适合增删操作
    // 2.对于下标循环一圈类似的问题，通过%可以很好地实现循环，而不需要我们自己构造循环链表；
    public int lastRemaining(int n, int m) {
        if (n < 1 || m < 1)
            return -1; //出错
        LinkedList<Integer> list = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            list.add(i);
        }
        int removeIndex = 0;
        while (list.size() > 1) {
            removeIndex = (removeIndex + m - 1) % list.size();//实现动态的取余来完成循环链表遍历
            list.remove(removeIndex);

        }
        return list.getFirst();
    }

    /*
     * 63. 股票的最大利润
     * 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
     * */
    //- 思路：遍历每一个数字，并保存之前最小的数字，两者差最大即为最大利润。
    public int maxProfit(int[] prices) {
        if (prices.length == 0 || prices == null) {
            return 0;
        }
        int min = prices[0];//买入价格最小值
        int max = 0;//最大利润
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < min) {//保存“之前”最小数字
                min = prices[i];
            } else if (prices[i] - min > max) {//计算差值再比较大小
                max = prices[i] - min;
            }
        }
        return max;
    }

    /*
     * 64. 求1+2+…+n
     * 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
     * */
    //- 思路：对于A && B，如果A为假，那么就不执行B了；而如果A为真，就会执行B。
    //　　　对于A || B，如果A为真，那么就会不执行B了；而如果A为假，就会执行B。
    //　　使用递归来代替循环，用逻辑运算符&&或者||来代替判断语句。
    //　　代码实现功能为：当n大于1时，和为f(n)=f(n-1)+n，n=1时，f(n)=1

    public int sumNums(int n) {
        int sum = n;
        boolean flag = (n > 1) && ((sum += sumNums(n - 1)) > 0);//判断语句要写完整,要完整写出(sum+=getSum(n-1))>0,要赋值给flag才算完整的语句
        //上面这句话相当于：
        //if(n>1)
        //   sum+=getSum(n-1);

        //也可以使用||来实现
        //boolean flag = (n==1) || ((sum+=getSum(n-1))>0);
        return sum;
    }

    /*
     * 66. 构建乘积数组
     * 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B 中的元素 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。

     * */
    public int[] constructArr(int[] a) {
        if (a == null || a.length == 0) {
            return new int[0];
        }
        int[] b = new int[a.length];
        b[0] = 1;
        for (int i = 1; i < a.length; i++) {
            b[i] = b[i - 1] * a[i - 1];//左半部分C[i]=C[i-1]*A[i-1]）
        }
        int[] c = new int[a.length];
        c[a.length-1] = 1;
        for (int i = a.length - 2; i >= 0; i--) {
            c[i] = c[i + 1] * a[i + 1];//右半部分D[i]=D[i+1]*A[i+1]）
        }
        for (int i = 0; i < a.length; i++) {
            a[i] = b[i] * c[i];
        }
        return a;

    }

    /*
    68 - II. 二叉树的最近公共祖先
    * */
    /*
    * 二叉树公共节点的三种情况：

    p 和 q 都在左子树 ( right == null 或 left != null)
    p 和 q 都在右子树 ( left == null 或 right !=null)
    p 和 q 一个在左子树 一个在右子树 那么当前节点为最近公共祖先
    情况1：如果右子树找不到 p 或 q 即(right==null)，那么说明 p 和 q 都在左子树上，返回 left

    情况2：如果左子树找不到 p 或 q 即(left==null)，那么说明 p 和 q 都在右子树上，返回 right

    情况3：如果上述情况都不符合，说明 p 和 q 分别在左子树和右子树，那么当前节点即为最近公共祖先，直接返回 root 即可。

    * */
    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        //返回节点存在情况
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor1(root.left, p, q);
        TreeNode right = lowestCommonAncestor1(root.right, p, q);
        //情况1：如果右子树找不到 p 或 q 即(right==null)，
        //那么说明 p 和 q 都在左子树上，返回 left

        //情况2：如果左子树找不到 p 或 q 即(right==null)，
        //那么说明 p 和 q 都在右子树上，返回 right

        //如果上述情况都不符合，说明 p 和 q 分别在左子树和右子树，
        //那么最近公共节点为当前节点
        //直接返回 root 即可
        return (right == null) ? left : (left == null) ? right : root;
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
        System.out.println(new testForCode().countDigitOne1(12));
        new testForCode().isStraight(new int[]{1, 2, 3, 4, 5});
        System.out.println(Arrays.toString(new testForCode().reversePrint2(ListNode1)));
        System.out.println(new testForCode().cuttingRope(10));
    }
}


