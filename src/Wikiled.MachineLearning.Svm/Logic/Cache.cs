using System;

namespace Wikiled.MachineLearning.Svm.Logic
{
    internal class Cache
    {
        private readonly int count;

        private readonly HeadT[] head;

        private readonly HeadT lruHead;

        private long size;

        public Cache(int count, long size)
        {
            this.count = count;
            this.size = size;
            head = new HeadT[this.count];
            for (int i = 0; i < this.count; i++)
            {
                head[i] = new HeadT(this);
            }

            this.size /= 4;
            this.size -= this.count * (16 / 4); // sizeof(head_t) == 16
            lruHead = new HeadT(this);
            lruHead.next = lruHead.prev = lruHead;
        }

        // request data [0,len)
        // return some position p where [p,len) need to be filled
        // (p >= len if nothing needs to be filled)
        // java: simulate pointer using single-element array
        public int GetData(int index, ref float[] data, int len)
        {
            HeadT h = head[index];
            if (h.len > 0)
            {
                LruDelete(h);
            }

            int more = len - h.len;

            if (more > 0)
            {
                // free old space
                while (size < more)
                {
                    HeadT old = lruHead.next;
                    LruDelete(old);
                    size += old.len;
                    old.data = null;
                    old.len = 0;
                }

                // allocate new space
                float[] newData = new float[len];
                if (h.data != null)
                {
                    Array.Copy(h.data, 0, newData, 0, h.len);
                }

                h.data = newData;
                size -= more;
                Swap(ref h.len, ref len);
            }

            LruInsert(h);
            data = h.data;
            return len;
        }

        public void SwapIndex(int i, int j)
        {
            if (i == j)
            {
                return;
            }

            if (head[i].len > 0)
            {
                LruDelete(head[i]);
            }

            if (head[j].len > 0)
            {
                LruDelete(head[j]);
            }

            Swap(ref head[i].data, ref head[j].data);
            Swap(ref head[i].len, ref head[j].len);
            if (head[i].len > 0)
            {
                LruInsert(head[i]);
            }

            if (head[j].len > 0)
            {
                LruInsert(head[j]);
            }

            if (i > j)
            {
                Swap(ref i, ref j);
            }

            for (HeadT h = lruHead.next; h != lruHead; h = h.next)
            {
                if (h.len > i)
                {
                    if (h.len > j)
                    {
                        Swap(ref h.data[i], ref h.data[j]);
                    }
                    else
                    {
                        // give up
                        LruDelete(h);
                        size += h.len;
                        h.data = null;
                        h.len = 0;
                    }
                }
            }
        }

        private static void LruDelete(HeadT h)
        {
            // delete from current location
            h.prev.next = h.next;
            h.next.prev = h.prev;
        }

        private static void Swap<T>(ref T lhs, ref T rhs)
        {
            T tmp = lhs;
            lhs = rhs;
            rhs = tmp;
        }

        private void LruInsert(HeadT h)
        {
            // insert to last position
            h.next = lruHead;
            h.prev = lruHead.prev;
            h.prev.next = h;
            h.next.prev = h;
        }
    }
}
