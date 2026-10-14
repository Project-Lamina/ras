.text
fld1
cvtsi2sdq	%rcx, %xmm0
sarw	$15, %ax
cmpw	$609, %di
notl	%eax
sbbl	%eax, %eax
bswap	%r10d
orw	$128, 40(%r8)
movhps	8(%rcx), %xmm0
comisd	%xmm0, %xmm1
ucomisd	%xmm1, %xmm0
andw	$-3520, %dx
testw	$1025, %ax
shrw	$6, %ax
notq	%rsi
addw	22(%rsi), %r8w
notrack jmp	*%rax
rolb	%cl, %sil
rep movsq
rep stosq
movdqu	208(%rdx), %xmm0
rolw	$8, %cx
lock orq	$0, (%rsp)
bsrq	%rdi, %rax
cvtsi2sdl	%edx, %xmm0
punpcklqdq	%xmm1, %xmm0
movd	4(%rsp), %xmm0
movd	8(%rsp), %xmm6
punpckldq	%xmm6, %xmm0
movd	12(%rax), %xmm1
movd	%edi, %xmm0
movd	%r14d, %xmm2
pshuflw	$0, %xmm2, %xmm0
movd	%xmm0, 412(%rbx)
movd	%r14d, %xmm3
movd	%xmm1, 412(%rbx)
movd	%r14d, %xmm4
pshufd	$225, %xmm0, %xmm3
movd	%xmm0, %eax
movd	%xmm4, %edx
movd	%xmm1, %edx
btl	%eax, %r14d
btq	%rdx, %rcx
subw	$204, %si
cltd
movd	%r14d, %xmm6
movd	%eax, %xmm1
movd	%r12d, %xmm0
movd	%eax, %xmm5
movd	12(%rbp), %xmm2
movd	%eax, %xmm3
movd	18(%r15), %xmm1
movd	%ebp, %xmm0
movd	%xmm0, 18(%r15)
notrack jmp	*%r8
notrack jmp	*%rcx
notrack jmp	*%rdx
adcl	$-1, 64(%rsp)
movd	32(%r12), %xmm0
movd	%ebp, %xmm1
movd	60(%rsp), %xmm2
movd	64(%rax), %xmm0
movd	.LC102(%rip), %xmm1
movd	%eax, %xmm0
movd	%xmm0, -4(%rdx)
movd	76(%rsp), %xmm1
movd	72(%rsp), %xmm2
movd	(%rdx), %xmm0
movd	%ebp, %xmm2
movd	%xmm0, %edx
movd	320(%rbp), %xmm0
movd	%esi, %xmm1
movd	%xmm0, %ecx
movd	%ecx, %xmm1
movd	54(%r9), %xmm0
movd	.LC176(%rip), %xmm1
movd	%xmm0, 54(%r9)
movd	72(%rbp), %xmm1
rep stosl
movd	%r10d, %xmm2
movd	100(%rsp), %xmm0
movd	320(%rbx), %xmm0
movd	%r12d, %xmm1
movd	8(%rsp), %xmm1
pcmpeqd	%xmm0, %xmm0
movd	44(%rdx), %xmm0
movd	52(%rbx), %xmm0
movd	%xmm0, 24(%rsp)
movd	%eax, %xmm2
movd	24(%rsp), %xmm0
movd	%xmm0, 148(%rsp)
movd	148(%rsp), %xmm0
movd	%xmm0, 52(%rbx)
movd	52(%rbx), %xmm1
movd	.LC176(%rip), %xmm0
movd	56(%rsp), %xmm5
movd	.LC428(%rip), %xmm0
movd	%xmm1, 49(%rsp)
movd	%edx, %xmm1
movd	%r14d, %xmm0
movd	32(%r15), %xmm0
movd	%xmm0, 44(%r15)
movd	84(%r12), %xmm0
movd	72(%r12), %xmm1
movd	4(%rsp), %xmm1
movd	32(%rsp), %xmm2
movd	4(%rsp), %xmm3
movd	%edx, %xmm4
movd	96(%rsp), %xmm0
movd	144(%r15), %xmm6
movd	64(%rsp), %xmm2
movd	%ebx, %xmm5
movd	%xmm4, %eax
movd	%xmm4, 40(%rsp)
movd	%eax, %xmm4
movd	%xmm6, 172(%rsp)
movd	%xmm1, 64(%rsp)
movd	%xmm6, %r8d
movd	%xmm1, 232(%rsp)
movd	%xmm7, 176(%rsp)
movd	%xmm7, 68(%rax)
movd	144(%r15), %xmm0
movd	%esi, %xmm2
movd	8(%rbx), %xmm0
movd	%edx, %xmm0
movd	%ecx, %xmm3
movd	%ebx, %xmm1
movd	108(%rax), %xmm0
movd	12(%rsp), %xmm3
movd	%xmm3, %ecx
movd	44(%rbp), %xmm1
movd	44(%rsp), %xmm0
movd	44(%rbx), %xmm4
movd	%xmm0, %r13d
movd	%ebp, %xmm7
movd	%xmm0, 4(%rax)
movd	80(%rsp), %xmm5
movd	%r15d, %xmm0
movd	%edi, %xmm2
movd	36(%rbx), %xmm2
movd	40(%rbx), %xmm0
movd	144(%rbx), %xmm1
movd	%r10d, %xmm3
movd	64(%rbx), %xmm0
movd	%eax, %xmm7
movd	16(%rdx), %xmm2
movd	%r12d, %xmm3
movd	%xmm2, %ecx
movd	260(%rsp), %xmm0
movd	192(%rsp), %xmm1
movd	%xmm0, 72(%rax)
movd	%ecx, %xmm2
movd	%xmm2, 24(%rsp)
movd	%xmm0, 54(%rbx)
movd	%xmm0, 54(%rdx)
movd	%r10d, %xmm0
flds	.LC1(%rip)
fmul	%st(0), %st
fmul	%st, %st(1)
fstp	%st(0)
fildq	-16(%rsp)
fmulp	%st, %st(1)
fstpl	-16(%rsp)
rorq	%rax
cvtsi2ssq	%rcx, %xmm0
psubd	%xmm3, %xmm1
psubd	%xmm2, %xmm1
pextrw	$0, %xmm1, %esi
pand	%xmm1, %xmm0
movlpd	(%rbx), %xmm0
fldl	(%rdx)
fldz
fcomip	%st(1), %st
fchs
fldl	.LC77(%rip)
flds	.LC78(%rip)
fxch	%st(1)
fcomi	%st(1), %st
fstp	%st(1)
fnstcw	142(%rsp)
fld	%st(0)
fldcw	140(%rsp)
fistpq	72(%rsp)
fldcw	142(%rsp)
fildq	80(%rsp)
fadds	.LC79(%rip)
fucomip	%st(1), %st
fldl	64(%rsp)
fstl	72(%rsp)
fstpt	96(%rsp)
fldt	96(%rsp)
fistpl	16(%rsp)
fildl	16(%rsp)
fsubrp	%st, %st(1)
fmul	%st(1), %st
fsub	%st, %st(1)
btcq	$63, %rax
fmull	.LC81(%rip)
faddl	72(%rsp)
fstpl	72(%rsp)
fldl	.LC69(%rip)
fld	%st(1)
fxch	%st(3)
fstp	%st(3)
fxch	%st(2)
fld	%st(2)
fmul	%st(2), %st
fstp	%st(2)
flds	.LC70(%rip)
fcomi	%st(2), %st
fldl	.LC82(%rip)
flds	.LC83(%rip)
fcomip	%st(4), %st
fmul	%st, %st(3)
flds	.LC71(%rip)
fmull	.LC84(%rip)
fldl	(%rax)
ud2
paddb	%xmm1, %xmm0
fildq	(%rsi)
fldl	(%rdi)
fildq	(%rdi)
fldl	(%rsi)
paddw	%xmm1, %xmm0
cwtl
fldl	-56(%rbp)
fildq	-56(%rbp)
fldl	80(%rsp)
btsl	%ecx, %eax
pextrw	$0, %xmm2, %ecx
pextrw	$1, %xmm0, %eax
pextrw	$0, %xmm0, %eax
pextrw	$0, %xmm1, %edx
movhlps	%xmm1, %xmm3
btsl	%r9d, %edx
movss	7(%r11), %xmm1
minsd	80(%rsp), %xmm0
paddb	%xmm0, %xmm1
btrq	$63, %rsi
btcq	$63, %rdi
btrq	$33, %rax
movsbw	(%rax), %dx
movsbw	%dl, %ax
btsq	$63, %r14
btrq	%rax, %rdx
psubd	%xmm0, %xmm2
movss	%xmm2, %xmm0
movhlps	%xmm0, %xmm2
shufpd	$1, %xmm0, %xmm0
btsl	%edx, %eax
btsq	$33, 48(%rax)
fildq	48(%rsp)
fldl	(%rcx)
movhlps	%xmm0, %xmm4
pinsrw	$0, 54(%rax), %xmm0
movq	%rbp, %xmm0
movq	%xmm0, %rcx
movq	%xmm2, %rdx
movq	%rcx, %xmm0
movq	%rax, %xmm1
movq	%xmm0, 20(%r12)
movq	%xmm0, -20(%rax)
movq	%rdx, %xmm1
movq	32(%rbx), %xmm0
movq	8(%rbp), %xmm2
movq	%xmm1, 8(%rbp)
movq	%xmm0, %rax
movq	(%r8), %xmm0
movq	(%rdi), %xmm0
movq	.LC47(%rip), %xmm1
movq	24(%r12), %xmm0
movq	%xmm0, 24(%r12)
movq	%rax, %xmm0
movq	8(%rbp), %xmm1
movq	%xmm0, 12(%rbp)
movq	%rbx, %xmm1
movq	(%rax,%rdx,8), %xmm0
movq	%rsi, %xmm0
movq	%rcx, %xmm1
movq	%rdx, %xmm0
movq	%rdx, %xmm2
movq	%rax, %xmm3
movq	%rax, %xmm2
movq	%r14, %xmm0
movq	%xmm0, (%rax)
movq	(%r12), %xmm0
movq	%rdx, %xmm3
movq	%rax, %xmm4
movq	%xmm0, 8(%r8)
movq	%xmm0, %r12
movq	%r15, %xmm0
movq	%r15, %xmm1
movq	%xmm2, %r15
movq	%xmm3, %r15
movq	%xmm0, %r15
movq	64(%rbx), %xmm0
movq	%xmm1, 4(%rax)
movq	%xmm0, 12(%r12)
movq	%rdi, %xmm0
movq	%rsi, %xmm1
movq	%rsi, %xmm2
movq	96(%rax), %xmm0
movq	%xmm0, 40(%rsp)
movq	%rax, %xmm7
movq	%xmm0, 168(%rsp)
movq	%xmm0, 16(%rsp)
movq	16(%rsp), %xmm0
movq	%xmm0, 64(%rsp)
movq	64(%rsp), %xmm0
movq	%xmm0, 4(%rax)
movq	%rdi, %xmm2
movq	%rdi, %xmm1
movq	%rcx, %xmm3
movq	%rcx, %xmm4
movq	%r12, %xmm0
movq	%xmm0, %r13
movq	%rbx, %xmm0
movq	28(%rbx), %xmm1
movq	.LC306(%rip), %xmm0
movq	%xmm0, 28(%rbx)
movq	%xmm0, 56(%rbx)
movq	%r13, %xmm0
movq	%xmm0, (%r9)
movq	(%rcx,%rax,8), %xmm0
movq	%xmm0, 92+pcache1_g(%rip)
movq	%xmm3, %rsi
movq	%xmm1, %rax
movq	8(%rbx), %xmm1
movq	%xmm0, 8(%r9)
movq	%r8, %xmm1
movq	%rax, %xmm5
movq	%rax, %xmm6
movq	%r15, %xmm2
movq	(%rdx,%rax), %xmm0
movq	(%rdx), %xmm0
movq	%xmm0, 12(%rbx)
movq	%xmm0, 88(%r13)
movq	16(%r12), %xmm0
movq	%r12, %xmm1
movq	%xmm0, 56(%rsp)
movq	56(%rsp), %xmm0
movq	%xmm0, %rdi
movq	%xmm0, 36(%r15)
movq	%r14, %xmm2
movq	%xmm0, 36(%rbp)
movq	%xmm2, %r12
movq	%xmm0, 48(%r11)
movq	%rcx, %xmm6
movq	%xmm0, 12(%rdi)
movq	%xmm0, 68(%rdi)
movq	%xmm5, %r12
movq	40(%rdi), %xmm0
movq	.LC524(%rip), %xmm5
movq	%xmm0, 40(%rdi)
movq	40(%rdi), %xmm1
movq	52(%r14), %xmm1
movq	.LC47(%rip), %xmm0
movq	%xmm0, 52(%r14)
movq	%xmm0, 88(%rsp)
movq	88(%rsp), %xmm0
movq	52(%rax), %xmm1
movq	%xmm0, 52(%rax)
movq	%xmm1, 68(%rdi)
movq	%xmm0, 100(%r12)
movq	%rsi, %xmm7
movq	%xmm1, 4(%rdx)
movq	%rdx, %xmm4
movq	%xmm0, 8(%rax)
movq	%xmm0, 172(%rax)
movq	%xmm0, 24(%rdx)
movq	%xmm0, 24(%rax)
movq	%rcx, %xmm2
movq	8(%rbp), %xmm0
movq	%xmm0, 8(%rbx)
movq	48(%rcx), %xmm0
movq	%rdx, %xmm7
movq	104(%rbx), %xmm3
movq	%xmm0, 88(%rax)
movq	48(%r12), %xmm0
movq	40(%rbp), %xmm0
movq	%xmm0, 96(%rbp)
movq	%rbp, %xmm7
movq	%r13, %xmm1
movq	24(%rdx,%rcx), %xmm0
movq	376(%rbx), %xmm0
movq	%xmm0, 176(%rsp)
movq	176(%rsp), %xmm0
movq	260(%rsp), %xmm0
movq	%xmm0, 12(%rax)
movq	%xmm1, 72(%rbx)
movq	%xmm1, 232(%rsp)
movq	232(%rsp), %xmm1
movq	.LC683(%rip), %xmm1
movq	%xmm0, 40(%rbp)
movq	%xmm0, %rsi
movq	%xmm1, %r12
movq	%xmm0, (%rsp)
movq	%xmm1, %rsi
movq	%xmm4, %rsi
movq	8(%rsp), %xmm1
movq	%xmm0, 4(%rcx)
sete 11(%rdi)
setp %al
setne %ah
setg %r9b
movabsq $-9223372036854775808, %rcx
cmovne 34(%rsp), %bp
movq 0, %rax
movzwl 94, %eax
movzbl -1, %eax
movdqa 0, %xmm2
pushq (%r14,%rax,8)
popq 8(%r12)
pushq $131072
pushq $-1
movq %xmm9, %xmm10
imulq %rbp
imull %r9d
imulq $1000, (%rsp), %rcx
imull $3, 12(%r12), %r9d
cvtss2sd %xmm1, %xmm9
fsub %st(2), %st
fdivp %st, %st(1)
fdivp %st, %st(2)
rep movsb
rep movsw
rep movsl
rep stosb
rep stosw
rep cmpsb
rep cmpsw
rep cmpsl
rep cmpsq
rep scasb
rep scasw
rep scasl
rep scasq
rep lodsb
rep lodsw
rep lodsl
rep lodsq
repz ret
repne scasb
notrack jmp *%r8
lock orq $0, (%rsp)
repz ret
lock addw $1, external(%rip)
notrack jmp *external(%rip)
movd %rax, %xmm0
movd %xmm0, %rax
movd %r9, %xmm10
movd %xmm10, %r9
