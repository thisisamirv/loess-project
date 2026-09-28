c Cleveland LOESS reference implementation.
c Original source code for the LOESS algorithm by William S. Cleveland.
c This file combines the reference procedures listed below.

c Source: bbox.f
      subroutine ehg126(d,n,vc,x,v,nvmax)
         integer d,execnt,i,j,k,n,nv,nvmax,vc
         real machin,alpha,beta,mu,t
         real v(nvmax,d),x(n,d)
         real r1mach
         external r1mach
         save machin,execnt
         data execnt /0/
         execnt=execnt+1
         if(execnt.eq.1)then
            machin=r1mach(2)
         end if
c     fill in vertices for bounding box of $x$
c     lower left, upper right
         do 3 k=1,d
            alpha=machin
            beta=-machin
            do 4 i=1,n
               t=x(i,k)
               alpha=min(alpha,t)
               beta=max(beta,t)
    4       continue
c        expand the box a little
            mu=0.005*max(beta-alpha,1.e-10*max(abs(alpha),abs(beta))+1.e-30
     $      )
            alpha=alpha-mu
            beta=beta+mu
            v(1,k)=alpha
            v(vc,k)=beta
    3    continue
c     remaining vertices
         do 5 i=2,vc-1
            j=i-1
            do 6 k=1,d
               v(i,k)=v(1+mod(j,2)*(vc-1),k)
               j=float(j)/2.
    6       continue
    5    continue
         return
      end

c Source: cpvert.f
      subroutine ehg125(p,nv,v,vhit,nvmax,d,k,t,r,s,f,l,u)
         logical i1,i2,match
         integer d,execnt,h,i,i3,j,k,m,mm,nv,nvmax,p,r,s
         integer f(r,0:1,s),l(r,0:1,s),u(r,0:1,s),vhit(nvmax)
         real t
         real v(nvmax,d)
         external ehg182
         save execnt
         data execnt /0/
         execnt=execnt+1
         h=nv
         do 3 i=1,r
            do 4 j=1,s
               h=h+1
               do 5 i3=1,d
                  v(h,i3)=v(f(i,0,j),i3)
    5          continue
               v(h,k)=t
c           check for redundant vertex
               match=.false.
               m=1
    6          if(.not.match)then
                  i1=(m.le.nv)
               else
                  i1=.false.
               end if
               if(.not.(i1))goto 7
               match=(v(m,1).eq.v(h,1))
               mm=2
    8          if(match)then
                  i2=(mm.le.d)
               else
                  i2=.false.
               end if
               if(.not.(i2))goto 9
               match=(v(m,mm).eq.v(h,mm))
               mm=mm+1
               goto 8
    9          m=m+1
               goto 6
    7          m=m-1
               if(match)then
                  h=h-1
               else
                  m=h
                  if(vhit(1).ge.0)then
                     vhit(m)=p
                  end if
               end if
               l(i,0,j)=f(i,0,j)
               l(i,1,j)=m
               u(i,0,j)=m
               u(i,1,j)=f(i,1,j)
    4       continue
    3    continue
         nv=h
         if(.not.(nv.le.nvmax))then
            call ehg182(180)
         end if
         return
      end

c Source: descend.f
      integer function ehg138(i,z,a,xi,lo,hi,ncmax)
         logical i1
         integer d,execnt,i,j,nc,ncmax
         integer a(ncmax),hi(ncmax),lo(ncmax)
         real xi(ncmax),z(8)
         save execnt
         data execnt /0/
         execnt=execnt+1
c     descend tree until leaf or ambiguous
         j=i
    3    if(a(j).ne.0)then
            i1=(z(a(j)).ne.xi(j))
         else
            i1=.false.
         end if
         if(.not.(i1))goto 4
         if(z(a(j)).le.xi(j))then
            j=lo(j)
         else
            j=hi(j)
         end if
         goto 3
    4    ehg138=j
         return
      end

c Source: ehg106.f
      subroutine ehg106(il,ir,k,nk,p,pi,n)
         integer execnt,i,ii,il,ir,j,k,l,n,nk,r
         integer pi(n)
         real t
         real p(nk,n)
         save execnt
         data execnt /0/
         execnt=execnt+1
c     Select the k-th smallest value.
c     Floyd+Rivest, CACM Mar '75, Algorithm 489
         l=il
         r=ir
    3    if(.not.(l.lt.r))goto 4
c     Use an iterative partition of the interval [l, r] about t.
         t=p(1,pi(k))
         i=l
         j=r
         ii=pi(l)
         pi(l)=pi(k)
         pi(k)=ii
         if(t.lt.p(1,pi(r)))then
            ii=pi(l)
            pi(l)=pi(r)
            pi(r)=ii
         end if
    5    if(.not.(i.lt.j))goto 6
         ii=pi(i)
         pi(i)=pi(j)
         pi(j)=ii
         i=i+1
         j=j-1
    7    if(.not.(p(1,pi(i)).lt.t))goto 8
         i=i+1
         goto 7
    8    continue
    9    if(.not.(t.lt.p(1,pi(j))))goto 10
         j=j-1
         goto 9
   10    goto 5
    6    if(p(1,pi(l)).eq.t)then
            ii=pi(l)
            pi(l)=pi(j)
            pi(j)=ii
         else
            j=j+1
            ii=pi(r)
            pi(r)=pi(j)
            pi(j)=ii
         end if
         if(j.le.k)then
            l=j+1
         end if
         if(k.le.j)then
            r=j-1
         end if
         goto 3
    4    return
      end

c Source: ehg127.f
      subroutine ehg127(q,n,d,nf,f,x,psi,y,rw,kernel,k,dist,eta,b,od,w,r
     $cond,sing,sigma,u,e,gamma,qraux,work,tol,dd,tdeg,cdeg,s)
         integer column,d,dd,execnt,i,i3,i9,info,inorm2,j,jj,jpvt,k,kernel,
     $   n,nf,od,sing,tdeg
         integer cdeg(8),psi(n)
         real machep,f,i1,i10,i2,i4,i5,i6,i7,i8,rcond,rho,scal,tol
         real g(15),sigma(15),u(15,15),e(15,15),b(nf,k),colnor(15),dist(n),
     $   eta(nf),gamma(15),q(d),qraux(15),rw(n),s(0:od),w(nf),work(15),x(n,
     $   d),y(n)
         external ehg106,ehg182,ehg184,sqrdc,sqrsl,ssvdc
         integer isamax
         external isamax
         real r1mach
         external r1mach
         real sdot
         external sdot
         save machep,execnt
         data execnt /0/
         execnt=execnt+1
         if(execnt.eq.1)then
            machep=r1mach(4)
         end if
c     sort by distance
         do 3 i3=1,n
            dist(i3)=0
    3    continue
         do 4 j=1,dd
            i4=q(j)
            do 5 i3=1,n
               dist(i3)=dist(i3)+(x(i3,j)-i4)**2
    5       continue
    4    continue
         call ehg106(1,n,nf,1,dist,psi,n)
         rho=dist(psi(nf))*max(1.,f)
         if(.not.(0.lt.rho))then
            call ehg182(120)
         end if
c     compute neighborhood weights
         if(kernel.eq.2)then
            do 6 i=1,nf
               if(dist(psi(i)).lt.rho)then
                  i1=sqrt(rw(psi(i)))
               else
                  i1=0
               end if
               w(i)=i1
    6       continue
         else
            do 7 i3=1,nf
               w(i3)=sqrt(dist(psi(i3))/rho)
    7       continue
            do 8 i3=1,nf
               w(i3)=sqrt(rw(psi(i3))*(1-w(i3)**3)**3)
    8       continue
         end if
         if(abs(w(isamax(nf,w,1))).eq.0)then
            call ehg184('at ',q,dd,1)
            call ehg184('radius ',rho,1,1)
            if(.not..false.)then
               call ehg182(121)
            end if
         end if
c     fill design matrix
         column=1
         do 9 i3=1,nf
            b(i3,column)=w(i3)
    9    continue
         if(tdeg.ge.1)then
            do 10 j=1,d
               if(cdeg(j).ge.1)then
                  column=column+1
                  i5=q(j)
                  do 11 i3=1,nf
                     b(i3,column)=w(i3)*(x(psi(i3),j)-i5)
   11             continue
               end if
   10       continue
         end if
         if(tdeg.ge.2)then
            do 12 j=1,d
               if(cdeg(j).ge.1)then
                  if(cdeg(j).ge.2)then
                     column=column+1
                     i6=q(j)
                     do 13 i3=1,nf
                        b(i3,column)=w(i3)*(x(psi(i3),j)-i6)**2
   13                continue
                  end if
                  do 14 jj=j+1,d
                     if(cdeg(jj).ge.1)then
                        column=column+1
                        i7=q(j)
                        i8=q(jj)
                        do 15 i3=1,nf
                           b(i3,column)=w(i3)*(x(psi(i3),j)-i7)*(x(psi(i3),
     $                     jj)-i8)
   15                   continue
                     end if
   14             continue
               end if
   12       continue
            k=column
         end if
         do 16 i3=1,nf
            eta(i3)=w(i3)*y(psi(i3))
   16    continue
c     equilibrate columns
         do 17 j=1,k
            scal=0
            do 18 inorm2=1,nf
               scal=scal+b(inorm2,j)**2
   18       continue
            scal=sqrt(scal)
            if(0.lt.scal)then
               do 19 i3=1,nf
                  b(i3,j)=b(i3,j)/scal
   19          continue
               colnor(j)=scal
            else
               colnor(j)=1
            end if
   17    continue
c     singular value decomposition
         call sqrdc(b,nf,nf,k,qraux,jpvt,work,0)
         call sqrsl(b,nf,nf,k,qraux,eta,work,eta,eta,work,work,1000,info)
         do 20 i9=1,k
            do 21 i3=1,k
               u(i3,i9)=0
   21       continue
   20    continue
         do 22 i=1,k
            do 23 j=i,k
               u(i,j)=b(i,j)
   23       continue
   22    continue
         call ssvdc(u,15,k,k,sigma,g,u,15,e,15,work,21,info)
         if(.not.(info.eq.0))then
            call ehg182(182)
         end if
         tol=sigma(1)*(100*machep)
         rcond=min(rcond,sigma(k)/sigma(1))
         if(sigma(k).le.tol)then
            sing=sing+1
            if(sing.eq.1)then
               call ehg184('Warning. pseudoinverse used at',q,d,1)
               call ehg184('neighborhood radius',sqrt(rho),1,1)
               call ehg184('reciprocal condition number ',rcond,1,1)
            else
               if(sing.eq.2)then
                  call ehg184('There are other near singularities as well.'
     $            ,rho,1,1)
               end if
            end if
         end if
c     compensate for equilibration
         do 24 j=1,k
            i10=colnor(j)
            do 25 i3=1,k
               e(j,i3)=e(j,i3)/i10
   25       continue
   24    continue
c     solve least squares problem
         do 26 j=1,k
            if(tol.lt.sigma(j))then
               i2=sdot(k,u(1,j),1,eta,1)/sigma(j)
            else
               i2=0.
            end if
            gamma(j)=i2
   26    continue
         do 27 j=0,od
c     Handle k = 1 when od > 1.
            if(j.lt.k)then
               s(j)=sdot(k,e(j+1,1),15,gamma,1)
            else
               s(j)=0.
            end if
   27    continue
         return
      end

c Source: ehg131.f
      subroutine ehg131(x,y,rw,trl,diagl,kernel,k,n,d,nc,ncmax,vc,nv,nvm
     $ax,nf,f,a,c,hi,lo,pi,psi,v,vhit,vval,xi,dist,eta,b,ntol,fd,w,vval2
     $,rcond,sing,dd,tdeg,cdeg,lq,lf,setlf)
         logical setlf
         integer identi,d,dd,execnt,i1,i2,j,k,kernel,n,nc,ncmax,nf,ntol,nv,
     $   nvmax,sing,tdeg,vc
         integer lq(nvmax,nf),a(ncmax),c(vc,ncmax),cdeg(8),hi(ncmax),lo(ncm
     $   ax),pi(n),psi(n),vhit(nvmax)
         real f,fd,rcond,trl
         real lf(0:d,nvmax,nf),b(*),delta(8),diagl(n),dist(n),eta(nf),rw(n)
     $   ,v(nvmax,d),vval(0:d,nvmax),vval2(0:d,nvmax),w(nf),x(n,d),xi(ncmax
     $   ),y(n)
         external ehg126,ehg182,ehg139,ehg124
         real snrm2
         external snrm2
         save execnt
         data execnt /0/
         execnt=execnt+1
         if(.not.(d.le.8))then
            call ehg182(101)
         end if
c     build $k$-d tree
         call ehg126(d,n,vc,x,v,nvmax)
         nv=vc
         nc=1
         do 3 j=1,vc
            c(j,nc)=j
            vhit(j)=0
    3    continue
         do 4 i1=1,d
            delta(i1)=v(vc,i1)-v(1,i1)
    4    continue
         fd=fd*snrm2(d,delta,1)
         do 5 identi=1,n
            pi(identi)=identi
    5    continue
         call ehg124(1,n,d,n,nv,nc,ncmax,vc,x,pi,a,xi,lo,hi,c,v,vhit,nvmax,
     $   ntol,fd,dd)
c     smooth
         if(trl.ne.0)then
            do 6 i2=1,nv
               do 7 i1=0,d
                  vval2(i1,i2)=0
    7          continue
    6       continue
         end if
         call ehg139(v,nvmax,nv,n,d,nf,f,x,pi,psi,y,rw,trl,kernel,k,dist,di
     $   st,eta,b,d,w,diagl,vval2,nc,vc,a,xi,lo,hi,c,vhit,rcond,sing,dd,tde
     $   g,cdeg,lq,lf,setlf,vval)
         return
      end

c Source: ehg133.f
      subroutine ehg133(n,d,vc,nvmax,nc,ncmax,a,c,hi,lo,v,vval,xi,m,z,s)
         integer d,execnt,i,i1,m,nc,ncmax,nv,nvmax,vc
         integer a(ncmax),c(vc,ncmax),hi(ncmax),lo(ncmax)
         real delta(8),s(m),v(nvmax,d),vval(0:d,nvmax),xi(ncmax),z(m,d)
         real ehg128
         external ehg128
         save execnt
         data execnt /0/
         execnt=execnt+1
         do 3 i=1,m
            do 4 i1=1,d
               delta(i1)=z(i,i1)
    4       continue
            s(i)=ehg128(delta,d,ncmax,vc,a,xi,lo,hi,c,v,nvmax,vval)
    3    continue
         return
      end

c Source: ehg140.f
      subroutine ehg140(iw,i,j)
         integer execnt,i,j
         integer iw(i)
         save execnt
         data execnt /0/
         execnt=execnt+1
         iw(i)=j
         return
      end

c Source: ehg141.f
      subroutine ehg141(trl,n,deg,k,d,nsing,dk,delta1,delta2)
         integer d,deg,dk,k,n,nsing
         external ehg176
         real ehg176
         real corx,delta1,delta2,trl,z
         real c(48), c1, c2, c3, c4
c     coef, d, deg, del
         data c /
     $    .2971620,.3802660,.5886043
     $   ,.4263766,.3346498,.6271053
     $   ,.5241198,.3484836,.6687687
     $   ,.6338795,.4076457,.7207693
     $   ,.1611761,.3091323,.4401023
     $   ,.2939609,.3580278,.5555741
     $   ,.3972390,.4171278,.6293196
     $   ,.4675173,.4699070,.6674802
     $   ,.2848308,.2254512,.2914126
     $   ,.5393624,.2517230,.3898970
     $   ,.7603231,.2969113,.4740130
     $   ,.9664956,.3629838,.5348889
     $   ,.2075670,.2822574,.2369957
     $   ,.3911566,.2981154,.3623232
     $   ,.5508869,.3501989,.4371032
     $   ,.7002667,.4291632,.4930370
     $    /
         if(deg.eq.0) dk=1
         if(deg.eq.1) dk=d+1
         if(deg.eq.2) dk=float((d+2)*(d+1))/2.
         corx=sqrt(k/float(n))
         z=(sqrt(k/trl)-corx)/(1-corx)
         if(nsing .eq. 0 .and. 1 .lt. z)
     $      call ehg184('Chernobyl! trL<k',trl,1,1)
         if(z .lt. 0) call ehg184('Chernobyl! trL>n',trl,1,1)
         z=min(1.0,max(0.0,z))
         c4=exp(ehg176(z))
         i=1+3*(min(d,4)-1+4*(deg-1))
         if(d.le.4)then
            c1=c(i)
            c2=c(i+1)
            c3=c(i+2)
         else
            c1=c(i)+(d-4)*(c(i)-c(i-3))
            c2=c(i+1)+(d-4)*(c(i+1)-c(i-2))
            c3=c(i+2)+(d-4)*(c(i+2)-c(i-1))
         endif
         delta1=n-trl*exp(c1*z**c2*(1-z)**c3*c4)
         i=i+24
         if(d.le.4)then
            c1=c(i)
            c2=c(i+1)
            c3=c(i+2)
         else
            c1=c(i)+(d-4)*(c(i)-c(i-3))
            c2=c(i+1)+(d-4)*(c(i+1)-c(i-2))
            c3=c(i+2)+(d-4)*(c(i+2)-c(i-1))
         endif
         delta2=n-trl*exp(c1*z**c2*(1-z)**c3*c4)
         return
      end

c Source: ehg167.f
      subroutine ehg167(iunit,d,vc,nc,nv,nvmax,v,a,xi,vval)
         integer iunit,d,vc,nc,nv,a(nc),magic,i,j
         real v(nvmax,d),xi(nc),vval(0:d,nv)
         write(iunit,*)d,nc,nv
         do 10 i=1,d
   10    write(iunit,*)v(1,i),v(vc,i)
         j = 0
         do 20 i=1,nc
            if(a(i).ne.0)then
               write(iunit,*)a(i),xi(i)
            else
               write(iunit,*)a(i),j
            end if
   20    continue
         do 30 i=1,nv
   30    write(iunit,*)(vval(j,i),j=0,d)
      end

c Source: ehg168.f
      subroutine ehg168(iunit,d,vc,nc,nv,nvmax,v,a,xi,vval)
         integer iunit,d,vc,nc,nv,a(nc),magic,i,j
         real v(nvmax,d),xi(nc),vval(0:d,nv)
         do 10 i=1,d
   10    read(iunit,*)v(1,i),v(vc,i)
         do 20 i=1,nc
   20    read(iunit,*)a(i),xi(i)
         do 30 i=1,nv
   30    read(iunit,*)(vval(j,i),j=0,d)
      end

c Source: ehg169.f
      subroutine ehg169(d,vc,nc,ncmax,nv,nvmax,v,a,xi,c,hi,lo)
         integer d,execnt,i,j,k,mc,mv,nc,ncmax,nv,nvmax,p,vc
         integer a(ncmax),c(vc,ncmax),hi(ncmax),lo(ncmax),novhit(1)
         real v(nvmax,d),xi(ncmax)
         external ehg125,ehg182
         integer ifloor
         external ifloor
         save execnt
         data execnt /0/
         execnt=execnt+1
c     remaining vertices
         do 3 i=2,vc-1
            j=i-1
            do 4 k=1,d
               v(i,k)=v(1+mod(j,2)*(vc-1),k)
               j=ifloor(float(j)/2.)
    4       continue
    3    continue
         mc=1
         mv=vc
         novhit(1)=-1
         do 5 j=1,vc
            c(j,mc)=j
    5    continue
         p=1
    6    if(.not.(p.le.nc))goto 7
         if(a(p).ne.0)then
            k=a(p)
c           left son
            mc=mc+1
            lo(p)=mc
c           right son
            mc=mc+1
            hi(p)=mc
            call ehg125(p,mv,v,novhit,nvmax,d,k,xi(p),2**(k-1),2**(d-k),
     $      c(1,p),c(1,lo(p)),c(1,hi(p)))
         end if
         p=p+1
         goto 6
    7    if(.not.(mc.eq.nc))then
            call ehg182(193)
         end if
         if(.not.(mv.eq.nv))then
            call ehg182(193)
         end if
         return
      end

c Source: ehg170.f
      subroutine ehg170(k,d,vc,nv,nvmax,nc,ncmax,a,c,hi,lo,v,vval,xi)
         integer d,execnt,i,j,nc,ncmax,nv,nvmax,vc
         integer a(ncmax),c(vc,ncmax),hi(ncmax),lo(ncmax)
         real v(nvmax,d),vval(0:d,nvmax),xi(ncmax)
         save execnt
         data execnt /0/
         execnt=execnt+1
         write(k,*)'      real function loeval(z)'
         write(k,50)d
         write(k,*)'      integer d,vc,nv,nc'
         write(k,51)nc,vc,nc
         write(k,52)nc,nc
         write(k,53)nv,d
         write(k,54)d,nv
         write(k,55)nc
         write(k,56)
         write(k,57)d,vc,nv,nc
   50    format('      real z(',i2,')')
   51    format('      integer a(',i5,'), c(',i3,',',i5,')')
   52    format('      integer hi(',i5,'), lo(',i5,')')
   53    format('      real v(',i5,',',i2,')')
   54    format('      real vval(0:',i2,',',i5,')')
   55    format('      real xi(',i5,')')
   56    format('      real ehg128')
   57    format('      data d,vc,nv,nc /',i2,',',i3,',',i5,',',i5,'/')
         do 3 i=1,nc
            write(k,58)i,a(i)
   58       format('      data a(',i5,') /',i5,'/')
            if(a(i).ne.0)then
               write(k,59)i,i,i,hi(i),lo(i),xi(i)
   59          format('      data hi(',i5,'),lo(',i5,'),xi(',i5,') /',
     $             i5,',',i5,',',1pe15.6,'/')
            end if
            do 4 j=1,vc
               write(k,60)j,i,c(j,i)
   60          format('      data c(',i3,',',i5,') /',i5,'/')
    4       continue
    3    continue
         do 5 i=1,nv
            write(k,61)i,vval(0,i)
   61       format('      data vval(0,',i5,') /',1pe15.6,'/')
            do 6 j=1,d
               write(k,62)i,j,v(i,j)
   62          format('      data v(',i5,',',i2,') /',1pe15.6,'/')
               write(k,63)j,i,vval(j,i)
   63          format('      data vval(',i2,',',i5,') /',1pe15.6,'/')
    6       continue
    5    continue
         write(k,*)'      loeval=ehg128(z,d,nc,vc,a,xi,lo,hi,c,v,nv,vval)'
         write(k,*)'      end'
         return
      end

c Source: ehg176.f
      real function ehg176(z)
         real z(*)
         integer d,vc,nv,nc
         integer a(17), c(2,17)
         integer hi(17), lo(17)
         real v(10,1)
         real vval(0:1,10)
         real xi(17)
         real ehg128
         data d,vc,nv,nc /1,2,10,17/
         data a(1) /1/
         data hi(1),lo(1),xi(1) /3,2,0.3705/
         data c(1,1) /1/
         data c(2,1) /2/
         data a(2) /1/
         data hi(2),lo(2),xi(2) /5,4,0.2017/
         data c(1,2) /1/
         data c(2,2) /3/
         data a(3) /1/
         data hi(3),lo(3),xi(3) /7,6,0.5591/
         data c(1,3) /3/
         data c(2,3) /2/
         data a(4) /1/
         data hi(4),lo(4),xi(4) /9,8,0.1204/
         data c(1,4) /1/
         data c(2,4) /4/
         data a(5) /1/
         data hi(5),lo(5),xi(5) /11,10,0.2815/
         data c(1,5) /4/
         data c(2,5) /3/
         data a(6) /1/
         data hi(6),lo(6),xi(6) /13,12,0.4536/
         data c(1,6) /3/
         data c(2,6) /5/
         data a(7) /1/
         data hi(7),lo(7),xi(7) /15,14,0.7132/
         data c(1,7) /5/
         data c(2,7) /2/
         data a(8) /0/
         data c(1,8) /1/
         data c(2,8) /6/
         data a(9) /0/
         data c(1,9) /6/
         data c(2,9) /4/
         data a(10) /0/
         data c(1,10) /4/
         data c(2,10) /7/
         data a(11) /0/
         data c(1,11) /7/
         data c(2,11) /3/
         data a(12) /0/
         data c(1,12) /3/
         data c(2,12) /8/
         data a(13) /0/
         data c(1,13) /8/
         data c(2,13) /5/
         data a(14) /0/
         data c(1,14) /5/
         data c(2,14) /9/
         data a(15) /1/
         data hi(15),lo(15),xi(15) /17,16,0.8751/
         data c(1,15) /9/
         data c(2,15) /2/
         data a(16) /0/
         data c(1,16) /9/
         data c(2,16) /10/
         data a(17) /0/
         data c(1,17) /10/
         data c(2,17) /2/
         data vval(0,1) /-9.0572E-2/
         data v(1,1) /-5.E-3/
         data vval(1,1) /4.4844/
         data vval(0,2) /-1.0856E-2/
         data v(2,1) /1.005/
         data vval(1,2) /-0.7736/
         data vval(0,3) /-5.3718E-2/
         data v(3,1) /0.3705/
         data vval(1,3) /-0.3495/
         data vval(0,4) /2.6152E-2/
         data v(4,1) /0.2017/
         data vval(1,4) /-0.7286/
         data vval(0,5) /-5.8387E-2/
         data v(5,1) /0.5591/
         data vval(1,5) /0.1611/
         data vval(0,6) /9.5807E-2/
         data v(6,1) /0.1204/
         data vval(1,6) /-0.7978/
         data vval(0,7) /-3.1926E-2/
         data v(7,1) /0.2815/
         data vval(1,7) /-0.4457/
         data vval(0,8) /-6.4170E-2/
         data v(8,1) /0.4536/
         data vval(1,8) /3.2813E-2/
         data vval(0,9) /-2.0636E-2/
         data v(9,1) /0.7132/
         data vval(1,9) /0.3350/
         data vval(0,10) /4.0172E-2/
         data v(10,1) /0.8751/
         data vval(1,10) /-4.1032E-2/
         ehg176=ehg128(z,d,nc,vc,a,xi,lo,hi,c,v,nv,vval)
      end

c Source: ehg182.f
      subroutine ehg182(i)
         integer i
         if(i.eq.100) print *,'wrong version number in lowesd.  Probably ty
     +po in caller.'
         if(i.eq.101) print *,'d>dMAX in ehg131.  Need to recompile with in
     +creased dimensions.'
         if(i.eq.102) print *,'liv too small.   (Discovered by lowesd)'
         if(i.eq.103) print *,'lv too small.    (Discovered by lowesd)'
         if(i.eq.104) print *,'alpha too small.  fewer data values than deg
     +rees of freedom.'
         if(i.eq.105) print *,'k>d2MAX in ehg136.  Need to recompile with i
     +ncreased dimensions.'
         if(i.eq.106) print *,'lwork too small'
         if(i.eq.107) print *,'invalid value for kernel'
         if(i.eq.108) print *,'invalid value for ideg'
         if(i.eq.109) print *,'lowstt only applies when kernel=1.'
         if(i.eq.110) print *,'not enough extra workspace for robustness ca
     +lculation'
         if(i.eq.120) print *,'zero-width neighborhood. make alpha bigger'
         if(i.eq.121) print *,'all data on boundary of neighborhood. make a
     +lpha bigger'
         if(i.eq.122) print *,'extrapolation not allowed with blending'
         if(i.eq.123) print *,'ihat=1 (diag L) in l2fit only makes sense if
     + z=x (eval=data).'
         if(i.eq.171) print *,'lowesd must be called first.'
         if(i.eq.172) print *,'lowesf must not come between lowesb and lowe
     +se, lowesr, or lowesl.'
         if(i.eq.173) print *,'lowesb must come before lowese, lowesr, or l
     +owesl.'
         if(i.eq.174) print *,'lowesb need not be called twice.'
         if(i.eq.180) print *,'nv>nvmax in cpvert.'
         if(i.eq.181) print *,'nt>20 in eval.'
         if(i.eq.182) print *,'svddc failed in l2fit.'
         if(i.eq.183) print *,'didnt find edge in vleaf.'
         if(i.eq.184) print *,'zero-width cell found in vleaf.'
         if(i.eq.185) print *,'trouble descending to leaf in vleaf.'
         if(i.eq.186) print *,'insufficient workspace for lowesf.'
         if(i.eq.187) print *,'insufficient stack space'
         if(i.eq.188) print *,'lv too small for computing explicit L'
         if(i.eq.191) print *,'computed trace L was negative; something is
     +wrong!'
         if(i.eq.192) print *,'computed delta was negative; something is wr
     +ong!'
         if(i.eq.193) print *,'workspace in loread appears to be corrupted'
         if(i.eq.194) print *,'trouble in l2fit/l2tr'
         if(i.eq.195) print *,'only constant, linear, or quadratic local mo
     +dels allowed'
         if(i.eq.196) print *,'degree must be at least 1 for vertex influen
     +ce matrix'
         if(i.eq.999) print *,'not yet implemented'
         print *,'Assert failed, error code ',i
         stop
      end
      subroutine ehg183(s,i,n,inc)
         character*(*) s
         integer n, inc, i(inc,n), j
         print *,s,(i(1,j),j=1,n)
      end
      subroutine ehg184(s,x,n,inc)
         character*(*) s
         integer n, inc, j
         real x(inc,n)
         print *,s,(x(1,j),j=1,n)
      end

c Source: ehg191.f
      subroutine ehg191(m,z,l,d,n,nf,nv,ncmax,vc,a,xi,lo,hi,c,v,nvmax,vv
     $al2,lf,lq)
         integer lq1,d,execnt,i,i1,i2,j,m,n,nc,ncmax,nf,nv,nvmax,p,vc
         integer lq(nvmax,nf),a(ncmax),c(vc,ncmax),hi(ncmax),lo(ncmax)
         real l(m,n),lf(0:d,nvmax,nf),v(nvmax,d),vval2(0:d,nvmax),xi(ncmax)
     $   ,z(m,d),zi(8)
         real ehg128
         external ehg128
         save execnt
         data execnt /0/
         execnt=execnt+1
         do 3 j=1,n
            do 4 i2=1,nv
               do 5 i1=0,d
                  vval2(i1,i2)=0
    5          continue
    4       continue
            do 6 i=1,nv
c           linear search for i in Lq
               lq1=lq(i,1)
               lq(i,1)=j
               p=nf
    7          if(.not.(lq(i,p).ne.j))goto 8
               p=p-1
               goto 7
    8          lq(i,1)=lq1
               if(lq(i,p).eq.j)then
                  do 9 i1=0,d
                     vval2(i1,i)=lf(i1,i,p)
    9             continue
               end if
    6       continue
            do 10 i=1,m
               do 11 i1=1,d
                  zi(i1)=z(i,i1)
   11          continue
               l(i,j)=ehg128(zi,d,ncmax,vc,a,xi,lo,hi,c,v,nvmax,vval2)
   10       continue
    3    continue
         return
      end

c Source: ehg196.f
      subroutine ehg196(tau,d,f,trl)
         integer d,dka,dkb,execnt,tau
         real alpha,f,trl,trla,trlb
         external ehg197
         save execnt
         data execnt /0/
         execnt=execnt+1
         call ehg197(1,tau,d,f,dka,trla)
         call ehg197(2,tau,d,f,dkb,trlb)
         alpha=float(tau-dka)/float(dkb-dka)
         trl=(1-alpha)*trla+alpha*trlb
         return
      end

c Source: ehg197.f
      subroutine ehg197(deg,tau,d,f,dk,trl)
         integer d,deg,dk,tau
         real trl, f
         dk = 0
         if(deg.eq.1) dk=d+1
         if(deg.eq.2) dk=float((d+2)*(d+1))/2.
         g1 = (-0.08125*d+0.13)*d+1.05
         trl = dk*(1+max(0.,(g1-f)/f))
         return
      end

c Source: eval.f
      real function ehg128(z,d,ncmax,vc,a,xi,lo,hi,c,v,nvmax,vval)
         logical i10,i2,i3,i4,i5,i6,i7,i8,i9
         integer d,execnt,i,i1,i11,i12,ig,ii,j,lg,ll,m,nc,ncmax,nt,nv,nvmax
     $   ,ur,vc
         integer a(ncmax),c(vc,ncmax),hi(ncmax),lo(ncmax),t(20)
         real ge,gn,gs,gw,gpe,gpn,gps,gpw,h,phi0,phi1,psi0,psi1,s,sew,sns,v
     $   0,v1,xibar
         real g(0:8,256),g0(0:8),g1(0:8),v(nvmax,d),vval(0:d,nvmax),xi(ncma
     $   x),z(d)
         external ehg182,ehg184
         save execnt
         data execnt /0/
         execnt=execnt+1
c     locate enclosing cell
         nt=1
         t(nt)=1
         j=1
    3    if(.not.(a(j).ne.0))goto 4
         nt=nt+1
c     Select the child cell that contains the evaluation point.
         if(z(a(j)).le.xi(j))then
            i1=lo(j)
         else
            i1=hi(j)
         end if
         t(nt)=i1
         if(.not.(nt.lt.20))then
            call ehg182(181)
         end if
         j=t(nt)
         goto 3
    4    continue
c     Evaluate the tensor-product interpolant.
         do 5 i12=1,vc
            do 6 i11=0,d
               g(i11,i12)=vval(i11,c(i12,j))
    6       continue
    5    continue
         lg=vc
         ll=c(1,j)
         ur=c(vc,j)
         do 7 i=d,1,-1
            h=(z(i)-v(ll,i))/(v(ur,i)-v(ll,i))
            if(h.lt.-.001)then
               call ehg184('eval ',z,d,1)
               call ehg184('lowerlimit ',v(ll,1),d,nvmax)
            else
               if(1.001.lt.h)then
                  call ehg184('eval ',z,d,1)
                  call ehg184('upperlimit ',v(ur,1),d,nvmax)
               end if
            end if
            if(-.001.le.h)then
               i2=(h.le.1.001)
            else
               i2=.false.
            end if
            if(.not.i2)then
               call ehg182(122)
            end if
            lg=float(lg)/2.
            do 8 ig=1,lg
c           Hermite basis
               phi0=(1-h)**2*(1+2*h)
               phi1=h**2*(3-2*h)
               psi0=h*(1-h)**2
               psi1=h**2*(h-1)
               g(0,ig)=phi0*g(0,ig)+phi1*g(0,ig+lg)+(psi0*g(i,ig)+psi1*g(i,
     $         ig+lg))*(v(ur,i)-v(ll,i))
               do 9 ii=1,i-1
                  g(ii,ig)=phi0*g(ii,ig)+phi1*g(ii,ig+lg)
    9          continue
    8       continue
    7    continue
         s=g(0,1)
c     Blend neighboring two-dimensional cells.
         if(d.eq.2)then
c        ----- North -----
            v0=v(ll,1)
            v1=v(ur,1)
            do 10 i11=0,d
               g0(i11)=vval(i11,c(3,j))
   10       continue
            do 11 i11=0,d
               g1(i11)=vval(i11,c(4,j))
   11       continue
            xibar=v(ur,2)
            m=nt-1
   12       if(m.eq.0)then
               i4=.true.
            else
               if(a(t(m)).eq.2)then
                  i3=(xi(t(m)).eq.xibar)
               else
                  i3=.false.
               end if
               i4=i3
            end if
            if(.not.(.not.i4))goto 13
            m=m-1
            goto 12
   13       if(m.ge.1)then
               m=hi(t(m))
   14          if(.not.(a(m).ne.0))goto 15
               if(z(a(m)).le.xi(m))then
                  m=lo(m)
               else
                  m=hi(m)
               end if
               goto 14
   15          if(v0.lt.v(c(1,m),1))then
                  v0=v(c(1,m),1)
                  do 16 i11=0,d
                     g0(i11)=vval(i11,c(1,m))
   16             continue
               end if
               if(v(c(2,m),1).lt.v1)then
                  v1=v(c(2,m),1)
                  do 17 i11=0,d
                     g1(i11)=vval(i11,c(2,m))
   17             continue
               end if
            end if
            h=(z(1)-v0)/(v1-v0)
c        Hermite basis
            phi0=(1-h)**2*(1+2*h)
            phi1=h**2*(3-2*h)
            psi0=h*(1-h)**2
            psi1=h**2*(h-1)
            gn=phi0*g0(0)+phi1*g1(0)+(psi0*g0(1)+psi1*g1(1))*(v1-v0)
            gpn=phi0*g0(2)+phi1*g1(2)
c        ----- South -----
            v0=v(ll,1)
            v1=v(ur,1)
            do 18 i11=0,d
               g0(i11)=vval(i11,c(1,j))
   18       continue
            do 19 i11=0,d
               g1(i11)=vval(i11,c(2,j))
   19       continue
            xibar=v(ll,2)
            m=nt-1
   20       if(m.eq.0)then
               i6=.true.
            else
               if(a(t(m)).eq.2)then
                  i5=(xi(t(m)).eq.xibar)
               else
                  i5=.false.
               end if
               i6=i5
            end if
            if(.not.(.not.i6))goto 21
            m=m-1
            goto 20
   21       if(m.ge.1)then
               m=lo(t(m))
   22          if(.not.(a(m).ne.0))goto 23
               if(z(a(m)).le.xi(m))then
                  m=lo(m)
               else
                  m=hi(m)
               end if
               goto 22
   23          if(v0.lt.v(c(3,m),1))then
                  v0=v(c(3,m),1)
                  do 24 i11=0,d
                     g0(i11)=vval(i11,c(3,m))
   24             continue
               end if
               if(v(c(4,m),1).lt.v1)then
                  v1=v(c(4,m),1)
                  do 25 i11=0,d
                     g1(i11)=vval(i11,c(4,m))
   25             continue
               end if
            end if
            h=(z(1)-v0)/(v1-v0)
c        Hermite basis
            phi0=(1-h)**2*(1+2*h)
            phi1=h**2*(3-2*h)
            psi0=h*(1-h)**2
            psi1=h**2*(h-1)
            gs=phi0*g0(0)+phi1*g1(0)+(psi0*g0(1)+psi1*g1(1))*(v1-v0)
            gps=phi0*g0(2)+phi1*g1(2)
c        ----- East -----
            v0=v(ll,2)
            v1=v(ur,2)
            do 26 i11=0,d
               g0(i11)=vval(i11,c(2,j))
   26       continue
            do 27 i11=0,d
               g1(i11)=vval(i11,c(4,j))
   27       continue
            xibar=v(ur,1)
            m=nt-1
   28       if(m.eq.0)then
               i8=.true.
            else
               if(a(t(m)).eq.1)then
                  i7=(xi(t(m)).eq.xibar)
               else
                  i7=.false.
               end if
               i8=i7
            end if
            if(.not.(.not.i8))goto 29
            m=m-1
            goto 28
   29       if(m.ge.1)then
               m=hi(t(m))
   30          if(.not.(a(m).ne.0))goto 31
               if(z(a(m)).le.xi(m))then
                  m=lo(m)
               else
                  m=hi(m)
               end if
               goto 30
   31          if(v0.lt.v(c(1,m),2))then
                  v0=v(c(1,m),2)
                  do 32 i11=0,d
                     g0(i11)=vval(i11,c(1,m))
   32             continue
               end if
               if(v(c(3,m),2).lt.v1)then
                  v1=v(c(3,m),2)
                  do 33 i11=0,d
                     g1(i11)=vval(i11,c(3,m))
   33             continue
               end if
            end if
            h=(z(2)-v0)/(v1-v0)
c        Hermite basis
            phi0=(1-h)**2*(1+2*h)
            phi1=h**2*(3-2*h)
            psi0=h*(1-h)**2
            psi1=h**2*(h-1)
            ge=phi0*g0(0)+phi1*g1(0)+(psi0*g0(2)+psi1*g1(2))*(v1-v0)
            gpe=phi0*g0(1)+phi1*g1(1)
c        ----- West -----
            v0=v(ll,2)
            v1=v(ur,2)
            do 34 i11=0,d
               g0(i11)=vval(i11,c(1,j))
   34       continue
            do 35 i11=0,d
               g1(i11)=vval(i11,c(3,j))
   35       continue
            xibar=v(ll,1)
            m=nt-1
   36       if(m.eq.0)then
               i10=.true.
            else
               if(a(t(m)).eq.1)then
                  i9=(xi(t(m)).eq.xibar)
               else
                  i9=.false.
               end if
               i10=i9
            end if
            if(.not.(.not.i10))goto 37
            m=m-1
            goto 36
   37       if(m.ge.1)then
               m=lo(t(m))
   38          if(.not.(a(m).ne.0))goto 39
               if(z(a(m)).le.xi(m))then
                  m=lo(m)
               else
                  m=hi(m)
               end if
               goto 38
   39          if(v0.lt.v(c(2,m),2))then
                  v0=v(c(2,m),2)
                  do 40 i11=0,d
                     g0(i11)=vval(i11,c(2,m))
   40             continue
               end if
               if(v(c(4,m),2).lt.v1)then
                  v1=v(c(4,m),2)
                  do 41 i11=0,d
                     g1(i11)=vval(i11,c(4,m))
   41             continue
               end if
            end if
            h=(z(2)-v0)/(v1-v0)
c        Hermite basis
            phi0=(1-h)**2*(1+2*h)
            phi1=h**2*(3-2*h)
            psi0=h*(1-h)**2
            psi1=h**2*(h-1)
            gw=phi0*g0(0)+phi1*g1(0)+(psi0*g0(2)+psi1*g1(2))*(v1-v0)
            gpw=phi0*g0(1)+phi1*g1(1)
c        Blend north and south edges.
            h=(z(2)-v(ll,2))/(v(ur,2)-v(ll,2))
c        Hermite basis
            phi0=(1-h)**2*(1+2*h)
            phi1=h**2*(3-2*h)
            psi0=h*(1-h)**2
            psi1=h**2*(h-1)
            sns=phi0*gs+phi1*gn+(psi0*gps+psi1*gpn)*(v(ur,2)-v(ll,2))
c        Blend east and west edges.
            h=(z(1)-v(ll,1))/(v(ur,1)-v(ll,1))
c        Hermite basis
            phi0=(1-h)**2*(1+2*h)
            phi1=h**2*(3-2*h)
            psi0=h*(1-h)**2
            psi1=h**2*(h-1)
            sew=phi0*gw+phi1*ge+(psi0*gpw+psi1*gpe)*(v(ur,1)-v(ll,1))
            s=(sns+sew)-s
         end if
         ehg128=s
         return
      end

c Source: f77.f
      integer function ifloor(x)
         real x
         ifloor=x
         if(ifloor.gt.x) ifloor=ifloor-1
      end
      real function sign(a1,a2)
         real a1, a2
         sign=abs(a1)
         if(a2.ge.0) sign=-sign
      end

c Source: l2fit.f
      subroutine ehg136(u,lm,m,n,d,nf,f,x,psi,y,rw,kernel,k,dist,eta,b,o
     $d,o,ihat,w,rcond,sing,dd,tdeg,cdeg,s)
         integer identi,d,dd,execnt,i,i1,ihat,info,j,k,kernel,l,lm,m,n,nf,o
     $   d,sing,tdeg
         integer cdeg(8),psi(n)
         real f,i2,rcond,scale,tol
         real o(m,n),sigma(15),e(15,15),g(15,15),b(nf,k),dist(n),eta(nf),ga
     $   mma(15),q(8),qraux(15),rw(n),s(0:od,m),u(lm,d),w(nf),work(15),x(n,
     $   d),y(n)
         external ehg127,ehg182,sqrsl
         real sdot
         external sdot
         save execnt
         data execnt /0/
         execnt=execnt+1
         if(.not.(k.le.nf-1))then
            call ehg182(104)
         end if
         if(.not.(k.le.15))then
            call ehg182(105)
         end if
         do 3 identi=1,n
            psi(identi)=identi
    3    continue
         do 4 l=1,m
            do 5 i1=1,d
               q(i1)=u(l,i1)
    5       continue
            call ehg127(q,n,d,nf,f,x,psi,y,rw,kernel,k,dist,eta,b,od,w,rcon
     $      d,sing,sigma,e,g,gamma,qraux,work,tol,dd,tdeg,cdeg,s(0,l))
            if(ihat.eq.1)then
c           Compute the diagonal of the influence matrix.
               if(.not.(m.eq.n))then
                  call ehg182(123)
               end if
c           Find the query point in the neighborhood ordering.
               i=1
    6          if(.not.(l.ne.psi(i)))goto 7
               i=i+1
               if(.not.(i.lt.nf))then
                  call ehg182(123)
               end if
               goto 6
    7          do 8 i1=1,nf
                  eta(i1)=0
    8          continue
               eta(i)=w(i)
c           Form the weighted basis vector.
               call sqrsl(b,nf,nf,k,qraux,eta,eta,eta,eta,eta,eta,1000,info
     $         )
c           Project eta into the singular-vector basis.
               do 9 i1=1,k
                  gamma(i1)=0
    9          continue
               do 10 j=1,k
                  i2=eta(j)
                  do 11 i1=1,k
                     gamma(i1)=gamma(i1)+i2*e(j,i1)
   11             continue
   10          continue
c           Apply inverse singular values.
               do 12 j=1,k
                  if(tol.lt.sigma(j))then
                     gamma(j)=gamma(j)/sigma(j)
                  else
                     gamma(j)=0.
                  end if
   12          continue
               o(l,1)=sdot(k,g(1,1),15,gamma,1)
            else
               if(ihat.eq.2)then
c              Compute the requested influence-matrix row.
                  do 13 i1=1,n
                     o(l,i1)=0
   13             continue
                  do 14 j=1,k
                     do 15 i1=1,nf
                        eta(i1)=0
   15                continue
                     do 16 i1=1,k
                        eta(i1)=e(i1,j)
   16                continue
                     call sqrsl(b,nf,nf,k,qraux,eta,eta,work,work,work,work
     $               ,10000,info)
                     if(tol.lt.sigma(j))then
                        scale=1./sigma(j)
                     else
                        scale=0.
                     end if
                     do 17 i1=1,nf
                        eta(i1)=eta(i1)*(scale*w(i1))
   17                continue
                     do 18 i=1,nf
                        o(l,psi(i))=o(l,psi(i))+g(1,j)*eta(i)
   18                continue
   14             continue
               end if
            end if
    4    continue
         return
      end

c Source: l2tr.f
      subroutine ehg139(v,nvmax,nv,n,d,nf,f,x,pi,psi,y,rw,trl,kernel,k,d
     $ist,phi,eta,b,od,w,diagl,vval2,ncmax,vc,a,xi,lo,hi,c,vhit,rcond,si
     $ng,dd,tdeg,cdeg,lq,lf,setlf,s)
         logical setlf
         integer identi,d,dd,execnt,i,i2,i3,i5,i6,ii,ileaf,info,j,k,kernel,
     $   l,n,nc,ncmax,nf,nleaf,nv,nvmax,od,sing,tdeg,vc
         integer lq(nvmax,nf),a(ncmax),c(vc,ncmax),cdeg(8),hi(ncmax),leaf(2
     $   56),lo(ncmax),phi(n),pi(n),psi(n),vhit(nvmax)
         real f,i1,i4,i7,rcond,scale,term,tol,trl
         real lf(0:d,nvmax,nf),sigma(15),u(15,15),e(15,15),b(nf,k),diagl(n)
     $   ,dist(n),eta(nf),gamma(15),q(8),qraux(15),rw(n),s(0:od,nv),v(nvmax
     $   ,d),vval2(0:d,nv),w(nf),work(15),x(n,d),xi(ncmax),y(n),z(8)
         external ehg127,ehg182,sqrsl,ehg137
         real ehg128
         external ehg128
         real sdot
         external sdot
         save execnt
         data execnt /0/
         execnt=execnt+1
c     l2fit with trace(L)
         if(.not.(k.le.nf-1))then
            call ehg182(104)
         end if
         if(.not.(k.le.15))then
            call ehg182(105)
         end if
         if(trl.ne.0)then
            do 3 i5=1,n
               diagl(i5)=0
    3       continue
            do 4 i6=1,nv
               do 5 i5=0,d
                  vval2(i5,i6)=0
    5          continue
    4       continue
         end if
         do 6 identi=1,n
            psi(identi)=identi
    6    continue
         do 7 l=1,nv
            do 8 i5=1,d
               q(i5)=v(l,i5)
    8       continue
            call ehg127(q,n,d,nf,f,x,psi,y,rw,kernel,k,dist,eta,b,od,w,rcon
     $      d,sing,sigma,u,e,gamma,qraux,work,tol,dd,tdeg,cdeg,s(0,l))
            if(trl.ne.0)then
c           invert $psi$
               do 9 i5=1,n
                  phi(i5)=0
    9          continue
               do 10 i=1,nf
                  phi(psi(i))=i
   10          continue
               do 11 i5=1,d
                  z(i5)=v(l,i5)
   11          continue
               call ehg137(z,vhit(l),leaf,nleaf,d,nv,nvmax,ncmax,vc,a,xi,lo
     $         ,hi,c,v)
               do 12 ileaf=1,nleaf
                  do 13 ii=lo(leaf(ileaf)),hi(leaf(ileaf))
                     i=phi(pi(ii))
                     if(i.ne.0)then
                        if(.not.(psi(i).eq.pi(ii)))then
                           call ehg182(194)
                        end if
                        do 14 i5=1,nf
                           eta(i5)=0
   14                   continue
                        eta(i)=w(i)
c                    $eta = Q sup T W e sub i$
                        call sqrsl(b,nf,nf,k,qraux,eta,work,eta,eta,work,wo
     $                  rk,1000,info)
                        do 15 j=1,k
                           if(tol.lt.sigma(j))then
                              i4=sdot(k,u(1,j),1,eta,1)/sigma(j)
                           else
                              i4=0.
                           end if
                           gamma(j)=i4
   15                   continue
                        do 16 j=1,d+1
c                       Handle k = 1 when od > 1.
                           if(j.le.k)then
                              vval2(j-1,l)=sdot(k,e(j,1),15,gamma,1)
                           else
                              vval2(j-1,l)=0
                           end if
   16                   continue
                        do 17 i5=1,d
                           z(i5)=x(pi(ii),i5)
   17                   continue
                        term=ehg128(z,d,ncmax,vc,a,xi,lo,hi,c,v,nvmax,vval2
     $                  )
                        diagl(pi(ii))=diagl(pi(ii))+term
                        do 18 i5=0,d
                           vval2(i5,l)=0
   18                   continue
                     end if
   13             continue
   12          continue
            end if
            if(setlf)then
c           Compute local influence weights.
               if(.not.(k.ge.d+1))then
                  call ehg182(196)
               end if
               do 19 i5=1,nf
                  lq(l,i5)=psi(i5)
   19          continue
               do 20 i6=1,nf
                  do 21 i5=0,d
                     lf(i5,l,i6)=0
   21             continue
   20          continue
               do 22 j=1,k
                  do 23 i5=1,nf
                     eta(i5)=0
   23             continue
                  do 24 i5=1,k
                     eta(i5)=u(i5,j)
   24             continue
                  call sqrsl(b,nf,nf,k,qraux,eta,eta,work,work,work,work,10
     $            000,info)
                  if(tol.lt.sigma(j))then
                     scale=1./sigma(j)
                  else
                     scale=0.
                  end if
                  do 25 i5=1,nf
                     eta(i5)=eta(i5)*(scale*w(i5))
   25             continue
                  do 26 i=1,nf
                     i7=eta(i)
                     do 27 i5=0,d
                        if(i5.lt.k)then
                           lf(i5,l,i)=lf(i5,l,i)+e(1+i5,j)*i7
                        else
                           lf(i5,l,i)=0
                        end if
   27                continue
   26             continue
   22          continue
            end if
    7    continue
         if(trl.ne.0)then
            if(n.le.0)then
               trl=0.
            else
               i3=n
               i1=diagl(i3)
               do 28 i2=i3-1,1,-1
                  i1=diagl(i2)+i1
   28          continue
               trl=i1
            end if
         end if
         return
      end

c Source: lofort.f
      subroutine lofort(iunit,iv,liv,lv,wv)
         integer execnt,iunit
         integer iv(*)
         real wv(*)
         external ehg170
         save execnt
         data execnt /0/
         execnt=execnt+1
         call ehg170(iunit,iv(2),iv(4),iv(6),iv(14),iv(5),iv(17),iv(iv(7)),
     $   iv(iv(8)),iv(iv(9)),iv(iv(10)),wv(iv(11)),wv(iv(13)),wv(iv(12)))
         return
      end

c Source: lohead.f
      subroutine lohead(iunit,d,vc,nc,nv)
         integer iunit,d,vc,nc,nv
         read(iunit,*)d,nc,nv
         vc = 2**d
      end

c Source: loread.f
      subroutine loread(iunit,d,vc,nc,nv,iv,liv,lv,v)
         integer bound,d,execnt,iunit,liv,lv,nc,nv,vc
         integer iv(liv)
         real v(lv)
         external ehg168,ehg169,ehg182
         save execnt
         data execnt /0/
         execnt=execnt+1
         iv(28)=173
         iv(2)=d
         iv(4)=vc
         iv(14)=nv
         iv(17)=nc
         iv(7)=50
         iv(8)=iv(7)+nc
         iv(9)=iv(8)+vc*nc
         iv(10)=iv(9)+nc
         bound=iv(10)+nc
         if(.not.(bound-1.le.liv))then
            call ehg182(102)
         end if
         iv(11)=50
         iv(13)=iv(11)+nv*d
         iv(12)=iv(13)+(d+1)*nv
         bound=iv(12)+nc
         if(.not.(bound-1.le.lv))then
            call ehg182(103)
         end if
         call ehg168(iunit,d,vc,nc,nv,nv,v(iv(11)),iv(iv(7)),v(iv(12)),v(iv
     $   (13)))
         call ehg169(d,vc,nc,nc,nv,nv,v(iv(11)),iv(iv(7)),v(iv(12)),iv(iv(8
     $   )),iv(iv(9)),iv(iv(10)))
         return
      end

c Source: losave.f
      subroutine losave(iunit,iv,liv,lv,v)
         integer execnt,iunit,liv,lv
         integer iv(liv)
         real v(lv)
         external ehg167
         save execnt
         data execnt /0/
         execnt=execnt+1
         call ehg167(iunit,iv(2),iv(4),iv(5),iv(6),iv(14),v(iv(11)),iv(iv(7
     $   )),v(iv(12)),v(iv(13)))
         return
      end

c Source: lowesa.f
      subroutine lowesa(trl,n,d,tau,nsing,delta1,delta2)
         integer d,dka,dkb,execnt,n,nsing,tau
         real alpha,d1a,d1b,d2a,d2b,delta1,delta2,trl
         external ehg141
         save execnt
         data execnt /0/
         execnt=execnt+1
         call ehg141(trl,n,1,tau,d,nsing,dka,d1a,d2a)
         call ehg141(trl,n,2,tau,d,nsing,dkb,d1b,d2b)
         alpha=float(tau-dka)/float(dkb-dka)
         delta1=(1-alpha)*d1a+alpha*d1b
         delta2=(1-alpha)*d2a+alpha*d2b
         return
      end

c Source: lowesb.f
      subroutine lowesb(xx,yy,ww,diagl,infl,iv,liv,lv,wv)
         logical infl,setlf
         integer execnt
         integer iv(*)
         real trl
         real diagl(*),wv(*),ww(*),xx(*),yy(*)
         external ehg131,ehg182,ehg183
         integer ifloor
         external ifloor
         save execnt
         data execnt /0/
         execnt=execnt+1
         if(.not.(iv(28).ne.173))then
            call ehg182(174)
         end if
         if(iv(28).ne.172)then
            if(.not.(iv(28).eq.171))then
               call ehg182(171)
            end if
         end if
         iv(28)=173
         if(infl)then
            trl=1.
         else
            trl=0.
         end if
         setlf=(iv(27).ne.iv(25))
         call ehg131(xx,yy,ww,trl,diagl,iv(20),iv(29),iv(3),iv(2),iv(5),iv(
     $   17),iv(4),iv(6),iv(14),iv(19),wv(1),iv(iv(7)),iv(iv(8)),iv(iv(9)),
     $   iv(iv(10)),iv(iv(22)),iv(iv(27)),wv(iv(11)),iv(iv(23)),wv(iv(13)),
     $   wv(iv(12)),wv(iv(15)),wv(iv(16)),wv(iv(18)),ifloor(iv(3)*wv(2)),wv
     $   (3),wv(iv(26)),wv(iv(24)),wv(4),iv(30),iv(33),iv(32),iv(41),iv(iv(
     $   25)),wv(iv(34)),setlf)
         if(iv(14).lt.iv(6)+float(iv(4))/2.)then
            call ehg183('Warning. k-d tree limited by memory; nvmax=',iv(14
     $      ),1,1)
         else
            if(iv(17).lt.iv(5)+2)then
               call ehg183('Warning. k-d tree limited by memory. ncmax=',iv
     $         (17),1,1)
            end if
         end if
         return
      end

c Source: lowesc.f
      subroutine lowesc(n,l,ll,trl,delta1,delta2)
         integer execnt,i,j,n
         real delta1,delta2,trl
         real l(n,n),ll(n,n)
         real sdot
         external sdot
         save execnt
         data execnt /0/
         execnt=execnt+1
c     Compute LL = (I - L) * transpose(I - L).
         do 3 i=1,n
            l(i,i)=l(i,i)-1
    3    continue
         do 4 i=1,n
            do 5 j=1,i
               ll(i,j)=sdot(n,l(i,1),n,l(j,1),n)
    5       continue
    4    continue
         do 6 i=1,n
            do 7 j=i+1,n
               ll(i,j)=ll(j,i)
    7       continue
    6    continue
         do 8 i=1,n
            l(i,i)=l(i,i)+1
    8    continue
c     Accumulate the first two traces.
         trl=0
         delta1=0
         do 9 i=1,n
            trl=trl+l(i,i)
            delta1=delta1+ll(i,i)
    9    continue
c     Compute delta2 as trace(LL squared).
         delta2=0
         do 10 i=1,n
            delta2=delta2+sdot(n,ll(i,1),n,ll(1,i),1)
   10    continue
         return
      end

c Source: lowesd.f
      subroutine lowesd(versio,iv,liv,lv,v,d,n,f,ideg,nvmax,setlf)
         logical setlf
         integer bound,d,execnt,i,i1,i2,ideg,j,liv,lv,n,ncmax,nf,nvmax,vc,v
     $   ersio
         integer iv(liv)
         real f
         real v(lv)
         external ehg182
         integer ifloor
         external ifloor
         save execnt
         data execnt /0/
         execnt=execnt+1
         if(.not.(versio.eq.106))then
            call ehg182(100)
         end if
         iv(28)=171
         iv(2)=d
         iv(3)=n
         vc=2**d
         iv(4)=vc
         if(.not.(0.lt.f))then
            call ehg182(120)
         end if
         nf=min(n,ifloor(n*f))
         iv(19)=nf
         iv(20)=1
         if(ideg.eq.0)then
            i1=1
         else
            if(ideg.eq.1)then
               i1=d+1
            else
               if(ideg.eq.2)then
                  i1=float((d+2)*(d+1))/2.
               end if
            end if
         end if
         iv(29)=i1
         iv(21)=1
         iv(14)=nvmax
         ncmax=nvmax
         iv(17)=ncmax
         iv(30)=0
         iv(32)=ideg
         if(.not.(ideg.ge.0))then
            call ehg182(195)
         end if
         if(.not.(ideg.le.2))then
            call ehg182(195)
         end if
         iv(33)=d
         do 3 i2=41,49
            iv(i2)=ideg
    3    continue
         iv(7)=50
         iv(8)=iv(7)+ncmax
         iv(9)=iv(8)+vc*ncmax
         iv(10)=iv(9)+ncmax
         iv(22)=iv(10)+ncmax
c     initialize permutation
         j=iv(22)-1
         do 4 i=1,n
            iv(j+i)=i
    4    continue
         iv(23)=iv(22)+n
         iv(25)=iv(23)+nvmax
         if(setlf)then
            iv(27)=iv(25)+nvmax*nf
         else
            iv(27)=iv(25)
         end if
         bound=iv(27)+n
         if(.not.(bound-1.le.liv))then
            call ehg182(102)
         end if
         iv(11)=50
         iv(13)=iv(11)+nvmax*d
         iv(12)=iv(13)+(d+1)*nvmax
         iv(15)=iv(12)+ncmax
         iv(16)=iv(15)+n
         iv(18)=iv(16)+nf
         iv(24)=iv(18)+iv(29)*nf
         iv(34)=iv(24)+(d+1)*nvmax
         if(setlf)then
            iv(26)=iv(34)+(d+1)*nvmax*nf
         else
            iv(26)=iv(34)
         end if
         bound=iv(26)+nf
         if(.not.(bound-1.le.lv))then
            call ehg182(103)
         end if
         v(1)=f
         v(2)=0.05
         v(3)=0.
         v(4)=1.
         return
      end

c Source: lowese.f
      subroutine lowese(iv,liv,lv,wv,m,z,s)
         integer execnt,m
         integer iv(*)
         real s(m),wv(*),z(m,1)
         external ehg133,ehg182
         save execnt
         data execnt /0/
         execnt=execnt+1
         if(.not.(iv(28).ne.172))then
            call ehg182(172)
         end if
         if(.not.(iv(28).eq.173))then
            call ehg182(173)
         end if
         call ehg133(iv(3),iv(2),iv(4),iv(14),iv(5),iv(17),iv(iv(7)),iv(iv(
     $   8)),iv(iv(9)),iv(iv(10)),wv(iv(11)),wv(iv(13)),wv(iv(12)),m,z,s)
         return
      end

c Source: lowesf.f
      subroutine lowesf(xx,yy,ww,iv,liv,lv,wv,m,z,l,ihat,s)
         logical i1
         integer execnt,ihat,m,n
         integer iv(*)
         real l(m,*),s(m),wv(*),ww(*),xx(*),yy(*),z(m,1)
         external ehg182,ehg136
         save execnt
         data execnt /0/
         execnt=execnt+1
         if(171.le.iv(28))then
            i1=(iv(28).le.174)
         else
            i1=.false.
         end if
         if(.not.i1)then
            call ehg182(171)
         end if
         iv(28)=172
         if(.not.(iv(14).ge.iv(19)))then
            call ehg182(186)
         end if
         call ehg136(z,m,m,iv(3),iv(2),iv(19),wv(1),xx,iv(iv(22)),yy,ww,iv(
     $   20),iv(29),wv(iv(15)),wv(iv(16)),wv(iv(18)),0,l,ihat,wv(iv(26)),wv
     $   (4),iv(30),iv(33),iv(32),iv(41),s)
         return
      end

c Source: lowesl.f
      subroutine lowesl(iv,liv,lv,wv,m,z,l)
         integer execnt,m,n
         integer iv(*)
         real l(m,*),wv(*),z(m,1)
         external ehg182,ehg191
         save execnt
         data execnt /0/
         execnt=execnt+1
         if(.not.(iv(28).ne.172))then
            call ehg182(172)
         end if
         if(.not.(iv(28).eq.173))then
            call ehg182(173)
         end if
         if(.not.(iv(26).ne.iv(34)))then
            call ehg182(175)
         end if
         call ehg191(m,z,l,iv(2),iv(3),iv(19),iv(6),iv(17),iv(4),iv(iv(7)),
     $   wv(iv(12)),iv(iv(10)),iv(iv(9)),iv(iv(8)),wv(iv(11)),iv(14),wv(iv(
     $   24)),wv(iv(34)),iv(iv(25)))
         return
      end

c Source: lowesp.f
      subroutine lowesp(n,y,yhat,pwgts,rwgts,pi,ytilde)
         integer identi,execnt,i2,i3,i5,m,n
         integer pi(n)
         real c,i1,i4,mad
         real pwgts(n),rwgts(n),y(n),yhat(n),ytilde(n)
         external ehg106
         integer ifloor
         external ifloor
         save execnt
         data execnt /0/
         execnt=execnt+1
c     median absolute deviation
         do 3 i5=1,n
            ytilde(i5)=abs(y(i5)-yhat(i5))*sqrt(pwgts(i5))
    3    continue
         do 4 identi=1,n
            pi(identi)=identi
    4    continue
         m=ifloor(float(n)/2.)+1
         call ehg106(1,n,m,1,ytilde,pi,n)
         if((n-m)+1.lt.m)then
            call ehg106(1,m-1,m-1,1,ytilde,pi,n)
            mad=(ytilde(pi(m-1))+ytilde(pi(m)))/2
         else
            mad=ytilde(pi(m))
         end if
c     magic constant
         c=(6*mad)**2/5
         do 5 i5=1,n
            ytilde(i5)=1-((y(i5)-yhat(i5))**2*pwgts(i5))/c
    5    continue
         do 6 i5=1,n
            ytilde(i5)=ytilde(i5)*sqrt(rwgts(i5))
    6    continue
         if(n.le.0)then
            i4=0.
         else
            i3=n
            i1=ytilde(i3)
            do 7 i2=i3-1,1,-1
               i1=ytilde(i2)+i1
    7       continue
            i4=i1
         end if
         c=n/i4
c     pseudovalues
         do 8 i5=1,n
            ytilde(i5)=yhat(i5)+(c*rwgts(i5))*(y(i5)-yhat(i5))
    8    continue
         return
      end

c Source: lowesw.f
      subroutine lowesw(res,n,rw,pi)
         integer identi,execnt,i,i1,n,nh
         integer pi(n)
         real cmad,rsmall
         real res(n),rw(n)
         external ehg106
         integer ifloor
         external ifloor
         real r1mach
         external r1mach
         save execnt
         data execnt /0/
         execnt=execnt+1
c     Translated from Devlin's Ratfor implementation.
c     find median of absolute residuals
         do 3 i1=1,n
            rw(i1)=abs(res(i1))
    3    continue
         do 4 identi=1,n
            pi(identi)=identi
    4    continue
         nh=ifloor(float(n)/2.)+1
c     partial sort to find 6*mad
         call ehg106(1,n,nh,1,rw,pi,n)
         if((n-nh)+1.lt.nh)then
            call ehg106(1,nh-1,nh-1,1,rw,pi,n)
            cmad=3*(rw(pi(nh))+rw(pi(nh-1)))
         else
            cmad=6*rw(pi(nh))
         end if
         rsmall=r1mach(1)
         if(cmad.lt.rsmall)then
            do 5 i1=1,n
               rw(i1)=1
    5       continue
         else
            do 6 i=1,n
               if(cmad*0.999.lt.rw(i))then
                  rw(i)=0
               else
                  if(cmad*0.001.lt.rw(i))then
                     rw(i)=(1-(rw(i)/cmad)**2)**2
                  else
                     rw(i)=1
                  end if
               end if
    6       continue
         end if
         return
      end

c Source: rbuild.f
      subroutine ehg124(ll,uu,d,n,nv,nc,ncmax,vc,x,pi,a,xi,lo,hi,c,v,vhi
     $t,nvmax,fc,fd,dd)
         logical i1,i2,i3,leaf
         integer d,dd,execnt,fc,i4,inorm2,k,l,ll,m,n,nc,ncmax,nv,nvmax,p,u,
     $   uu,vc,lower,upper,check,offset
         integer a(ncmax),c(vc,ncmax),hi(ncmax),lo(ncmax),pi(n),vhit(nvmax)
         real diam,fd
         real diag(8),sigma(8),v(nvmax,d),x(n,d),xi(ncmax)
         external ehg125,ehg106,ehg129
         integer isamax
         external isamax
         save execnt
         data execnt /0/
         execnt=execnt+1
         p=1
         l=ll
         u=uu
         lo(p)=l
         hi(p)=u
    3    if(.not.(p.le.nc))goto 4
         do 5 i4=1,dd
            diag(i4)=v(c(vc,p),i4)-v(c(1,p),i4)
    5    continue
         diam=0
         do 6 inorm2=1,dd
            diam=diam+diag(inorm2)**2
    6    continue
         diam=sqrt(diam)
         if((u-l)+1.le.fc)then
            i1=.true.
         else
            i1=(diam.le.fd)
         end if
         if(i1)then
            leaf=.true.
         else
            if(ncmax.lt.nc+2)then
               i2=.true.
            else
               i2=(nvmax.lt.nv+float(vc)/2.)
            end if
            leaf=i2
         end if
         if(.not.leaf)then
            call ehg129(l,u,dd,x,pi,n,sigma)
            k=isamax(dd,sigma,1)
            m=float(l+u)/2.
            call ehg106(l,u,m,1,x(1,k),pi,n)

c     Shift the split point past duplicate coordinate values.
            offset = 0
    7       if(((m+offset).ge.u).or.((m+offset).lt.l))then
               goto 8
            else
               if(offset.lt.0)then
                  lower = l
                  check = m + offset
                  upper = check
               else
                  lower = m + offset + 1
                  check = lower
                  upper = u
               end if
               call ehg106(lower,upper,check,1,x(1,k),pi,n)
               if(x(pi(m + offset),k).eq.x(pi(m+offset+1),k))then
                  offset = (-offset)
                  if(offset.ge.0)then
                     offset = offset + 1
                  end if
                  goto 7
               else
                  m = m + offset
                  goto 8
               end if
            end if

    8       if(v(c(1,p),k).eq.x(pi(m),k))then
               leaf=.true.
            else
               leaf=(v(c(vc,p),k).eq.x(pi(m),k))
            end if
         end if
         if(leaf)then
            a(p)=0
         else
            a(p)=k
            xi(p)=x(pi(m),k)
c           left son
            nc=nc+1
            lo(p)=nc
            lo(nc)=l
            hi(nc)=m
c           right son
            nc=nc+1
            hi(p)=nc
            lo(nc)=m+1
            hi(nc)=u
            call ehg125(p,nv,v,vhit,nvmax,d,k,xi(p),2**(k-1),2**(d-k),c(
     $      1,p),c(1,lo(p)),c(1,hi(p)))
         end if
         p=p+1
         l=lo(p)
         u=hi(p)
         goto 3
    4    return
      end

c Source: spread.f
      subroutine ehg129(l,u,d,x,pi,n,sigma)
         integer d,execnt,i,k,l,n,u
         integer pi(n)
         real machin,alpha,beta,t
         real sigma(d),x(n,d)
         real r1mach
         external r1mach
         save machin,execnt
         data execnt /0/
         execnt=execnt+1
         if(execnt.eq.1)then
            machin=r1mach(2)
         end if
         do 3 k=1,d
            alpha=machin
            beta=-machin
            do 4 i=l,u
               t=x(pi(i),k)
               alpha=min(alpha,x(pi(i),k))
               beta=max(beta,t)
    4       continue
            sigma(k)=beta-alpha
    3    continue
         return
      end

c Source: vleaf.f
      subroutine ehg137(z,kappa,leaf,nleaf,d,nv,nvmax,ncmax,vc,a,xi,lo,h
     $i,c,v)
         integer d,execnt,nc,ncmax,nleaf,p,stackt
         integer a(ncmax),hi(ncmax),leaf(256),lo(ncmax),pstack(20)
         real xi(ncmax),z(d)
         external ehg182
         save execnt
         data execnt /0/
         execnt=execnt+1
c     find leaf cells affected by $z$
         stackt=0
         p=1
         nleaf=0
    3    if(.not.(0.lt.p))goto 4
         if(a(p).eq.0)then
c           leaf
            nleaf=nleaf+1
            leaf(nleaf)=p
c           Pop
            if(stackt.ge.1)then
               p=pstack(stackt)
            else
               p=0
            end if
            stackt=max(0,stackt-1)
         else
            if(z(a(p)).eq.xi(p))then
c              Push
               stackt=stackt+1
               if(.not.(stackt.le.20))then
                  call ehg182(187)
               end if
               pstack(stackt)=hi(p)
               p=lo(p)
            else
               if(z(a(p)).le.xi(p))then
                  p=lo(p)
               else
                  p=hi(p)
               end if
            end if
         end if
         goto 3
    4    if(.not.(nleaf.le.256))then
            call ehg182(185)
         end if
         return
      end
