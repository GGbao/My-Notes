import java.util.*;

public class treeDemo {
    public static class TreeNode<T> {
        public T val;
        public TreeNode<T> left;
        public TreeNode<T> right;

        public TreeNode(T val) {
            this.val = val;
            this.left = null;
            this.right = null;
        }
    }



    /**
     * 前序（递归，非递归），中序（递归，非递归），后序（递归，非递归），层序
     */
    //************前序**************
    public List<Integer> preOrder(TreeNode<Integer> node) {
        List<Integer> list = new ArrayList<>();
        if (node == null) {
            return list;
        }
        list.add(node.val);
        list.addAll(preOrder(node.left));
        list.addAll(preOrder(node.right));
        return list;
    }

    //前序非遍历
    public List<Integer> preOrderIn(TreeNode<Integer> node) {
        //stack栈顶元素永远为cur父节点
        Stack<TreeNode<Integer>> stack = new Stack<>();
        TreeNode<Integer> cur = node;
        List<Integer> list = new ArrayList<>();
        if (cur == null) {
            return list;
        }
        //先到最左子节点
        while (cur != null || !stack.isEmpty()) {
            if (cur != null) {
                list.add(cur.val);
                stack.push(cur);
                cur = cur.left;
            } else {
                cur = stack.pop().right;
            }
        }
        return list;
    }

    //*************中序*****************
    public List<Integer> inOrder(TreeNode<Integer> node) {
        List<Integer> list = new ArrayList<>();
        if (node == null) {
            return list;
        }
        list.addAll(inOrder(node.left));
        list.add(node.val);
        list.addAll(inOrder(node.right));
        return list;
    }

    //中序非递归
    public List<Integer> inOrderIn(TreeNode<Integer> node) {
        //stack栈顶元素永远为cur父节点
        Stack<TreeNode<Integer>> stack = new Stack<>();
        TreeNode<Integer> cur = node;
        List<Integer> list = new ArrayList<>();
        if (cur == null) {
            return list;
        }
        while (cur != null || !stack.isEmpty()) {
            //先到最左子节点
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



    //**************后序********************
    public List<Integer> postOrder(TreeNode<Integer> node) {
        List<Integer> list = new ArrayList<>();
        if (node == null) {
            return list;
        }
        list.addAll(postOrder(node.left));
        list.addAll(postOrder(node.right));
        list.add(node.val);
        return list;
    }

    //后序非遍历
    public List<Integer> postOrderIn(TreeNode<Integer> node) {
        //stack栈顶元素永远为cur父节点
        Stack<TreeNode<Integer>> stack = new Stack<>();
        //cur:当前访问节点，pCur:上次访问节点
        TreeNode<Integer> cur = node;
        TreeNode<Integer> pCur = null;
        TreeNode<Integer> top = null;

        List<Integer> list = new ArrayList<>();
        if (cur == null) {
            return list;
        }
        while (cur != null || !stack.isEmpty()) {
            while (cur != null) {
                //先到最左子节点
                stack.push(cur);
                cur = cur.left;
            }
            //走到这里cur都为空，并且已经遍历到左端最底部
            //从栈中看一下栈顶元素（只看一眼，用top指针记下，先不出栈）
            top = stack.peek();

            //一个根节点被访问的前提：无右子树或右子树已经被访问过
            if (top.right == null || top.right == pCur) {
                //添加并弹出top
                list.add(top.val);
                stack.pop();
                //修改最近被访问的节点
                pCur = top;
            }
            /*这里的else语句可换成带条件的else if:
		else if (top->lchild == pCur)//若左子树刚被访问过，则需先进入右子树(根节点需再次入栈)
		因为：上面的条件没通过就一定是下面的条件满足。
		*/
            else {
                //不满足添加根节点
                //进入右子树，且肯定右子树不为空
                cur = top.right;

            }
        }
        return list;
    }

    //层序遍历
    public static List<Integer> levelOrder(TreeNode<Integer> node){
        Queue<TreeNode<Integer>> queue = new ArrayDeque<>();
        TreeNode<Integer> front = node;
        List<Integer> list = new ArrayList<>();
        if (node == null) {
            return list;
        }
        queue.add(node);
        while (!queue.isEmpty()) {
            front = queue.poll();
            list.add(front.val);
            if (front.left != null) {
                queue.add(front.left);
            }
            if (front.right != null) {
                queue.add(front.right);
            }
        }
        return list;
    }




    public static void main(String[] args) {
        //            1
        //              \
        //               2
        //              /
        //             3
        //preOr->123  in->132   post->321  level->123
        TreeNode<Integer> root = new TreeNode<Integer>(1);
        root.right = new TreeNode<Integer>(2);
        root.right.left = new TreeNode<Integer>(3);
        System.out.println(Arrays.asList(levelOrder(root)));

    }
}
