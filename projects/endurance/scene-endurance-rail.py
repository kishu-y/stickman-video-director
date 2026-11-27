"""Native portrait staging for Endurance."""
from functools import lru_cache
from pathlib import Path
import math
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter

ROOT=Path(__file__).parent

@lru_cache
def art(name):
    return Image.open(ROOT/'art'/'endurance'/f'{name}.png').convert('RGBA')

def ease(t):
    x=min(1,max(0,t))
    return x*x*(3-2*x)

def plate(name,size,z,fx,fy):
    image=art(name)
    w,h=size
    scale=max(w/image.width,h/image.height)*z
    x=max(0,min(image.width-w/scale,fx*image.width-w/2/scale))
    y=max(0,min(image.height-h/scale,fy*image.height-h/2/scale))
    return image.transform(size,Image.Transform.AFFINE,(1/scale,0,x,0,1/scale,y),Image.Resampling.BILINEAR)

@lru_cache
def seeds():
    rng=random.Random(1915)
    return [(rng.random(),rng.random(),rng.uniform(.4,1.3)) for _ in range(56)]

def flakes(im,t,under=False):
    w,h=im.size
    layer=Image.new('RGBA',im.size)
    pen=ImageDraw.Draw(layer)
    for x,y,v in seeds()[::2]:
        px=((x+t*.12*v)%1)*w
        py=((y+t*(.033 if not under else -.017)*v)%1)*h
        if under:
            pen.ellipse((px,py,px+v*3,py+v*3),fill=(152,201,210,85))
        else:
            pen.line((px,py,px+v*8,py+v*3),fill=(224,238,243,68),width=max(1,round(w/480)))
    im.alpha_composite(layer)

@lru_cache
def boat(w):
    im=art('boat')
    return im.resize((w,round(w*im.height/im.width)),Image.Resampling.LANCZOS)

@lru_cache
def title_font(w):
    return ImageFont.truetype(str(ROOT/'fonts/ArchivoBlack.ttf'),max(15,round(w*.053)))

def marker(im,text):
    w,h=im.size
    draw=ImageDraw.Draw(im)
    draw.text((w*.50,h*.145),text,font=title_font(w),anchor='mm',fill='#fff2ce',stroke_width=max(2,w//180),stroke_fill='#061725')

@lru_cache
def fitted(name,width):
    source=art(name)
    return source.resize((width,round(width*source.height/source.width)),Image.Resampling.LANCZOS)

def place(im,name,x,y,width,angle=0):
    sprite=fitted(name,max(1,round(width)))
    if angle:
        sprite=sprite.rotate(angle,Image.Resampling.BICUBIC,expand=True)
    im.alpha_composite(sprite,(round(x-sprite.width/2),round(y-sprite.height/2)))

def collision(size,t,mode):
    w,h=size
    if mode=='crack' and t<.42:
        return plate('trapped',size,1.12,.50,.50)
    im=plate('water',size,1.13,.48,.45)
    im=Image.blend(im,Image.new('RGBA',size,'#19374b'),.30)
    press=ease((t-.10)/1.35) if mode!='alarm' else 1
    snap=ease((t-1.31)/.60) if mode!='alarm' else 1
    sinking=ease(t/2.66) if mode=='sink' else 0
    if mode in ('break','sink'):
        press=max(.70,press)
        snap=max(.45,snap)
    y=(.535+.13*sinking)*h
    if snap==0:
        place(im,'solid-hull',.50*w,y,.90*w,0)
    else:
        # Before separation both halves occupy their original pixel position.
        width=round(.90*w)
        source=art('solid-hull')
        origin=round(.5*w-width/2)
        for name,side in [('hull-left',-1),('hull-right',1)]:
            raw=art(name)
            x0=0 if side<0 else source.width-raw.width
            cx=origin+(x0+raw.width/2)*width/source.width
            place(im,name,cx+side*.035*w*snap,y+side*.014*h*snap,
                  raw.width*width/source.width,-side*8*snap-16*sinking)
        if mode in ('crack','break') and 0<snap<1:
            for i,offset in enumerate((-.11,-.022,.075,.13)):
                place(im,'splinter',w*(.50+offset*snap),
                      h*(.49-(.13-i*.021)*math.sin(math.pi*snap*.88)),
                      w*(.018+i*.003),i*33+snap*45)
    rim=Image.new('RGBA',size)
    for name,x,y,scale,angle in [
        ('ice-left',(.01+.245*press)*w,.63*h,.63*w,-2-7*snap),
        ('ice-right',(.99-.245*press)*w,.66*h,.64*w,3+9*snap),
    ]:
        place(rim,name,x,y,scale,angle)
    im.alpha_composite(rim)
    if mode!='alarm':
        # Keep the foreground water in one coordinate system as the hull travels under it.
        water=plate('water',size,1.17,.52,.50)
        line=round(h*((.67-.06*sinking) if mode=='sink' else .748))
        mask=Image.new('L',size)
        ImageDraw.Draw(mask).rectangle((0,line,w,h),fill=255)
        im=Image.composite(water,im,mask.filter(ImageFilter.GaussianBlur(max(2,w*.012))))
    return im

def explorer(t,c,w,h,dt,shot):
    if not hasattr(c,'endurance_actor'):
        c.endurance_actor=c.puppet_runtime.Puppet(c.character_packages['explorer'],'three_quarter','explorer')
    p=c.endurance_actor
    shock=shot=='alarm'
    strike=ease((dt-.29)/.43)
    recover=ease((dt-.74)/1.02)
    brace=strike*(1-.82*recover) if shock else 0
    gesture=ease(dt/.76)*(1-.85*ease((dt-1.45)/.9)) if not shock else 0
    scale=w/720*(1.02 if shock else .91)
    theta=math.radians(-7*brace+1*gesture)
    x=.49*w+scale*306*math.sin(theta)
    y=.575*h+scale*(306*(1-math.cos(theta))+2*brace)
    xf=np.array([[scale*math.cos(theta),-scale*math.sin(theta),x],
                 [scale*math.sin(theta),scale*math.cos(theta),y],[0,0,1.]])
    feet=None
    if shock:
        feet={name:[.49*w+scale*ax,.575*h+scale*306] for name,ax in [('left',-53),('right',60)]}
    return p.pose(placement=xf,frame_index=round(t*30),ankle_targets=feet,
                  arm_actions={'right':dict(action='reach',phase=max(brace,gesture)),
                               'left':dict(action='reach',phase=brace)},
                  allow_cut=round(t*30) in (90,420,900),
                  head_variant='shock' if shock and .51<=dt else None,
                  head_angle_deg=-5*brace+3*gesture)

def rail_action(t,c,w,h,dt):
    if not hasattr(c,'endurance_actor'):
        c.endurance_actor=c.puppet_runtime.Puppet(c.character_packages['explorer'],'three_quarter','explorer')
    puppet=c.endurance_actor
    runtime=c.puppet_runtime
    scale=w/720
    home=np.array([.43*w,.575*h])
    arm=puppet.character.views['three_quarter'].arms['right']
    final_wrist=runtime.forward_kinematics(arm,runtime.PoseLibrary.sample(arm,'reach',1.0))[2]
    grip=home+scale*final_wrist
    reach=ease((dt-.22)/.73)
    shoulder_pull=ease((dt-.95)/.69)*(1-.43*ease((dt-1.83)/.61))
    turn=ease((dt-1.70)/.65)
    angle=math.radians(-1.65*shoulder_pull)
    rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    # Rotate about the fixed contact after the hand closes.
    origin=grip-scale*rot@final_wrist
    matrix=np.array([[scale*rot[0,0],scale*rot[0,1],origin[0]],
                     [scale*rot[1,0],scale*rot[1,1],origin[1]],[0,0,1.]])
    stance={name:list(home+scale*np.array([ax,306])) for name,ax in [('left',-53),('right',60)]}
    actor=puppet.pose(placement=matrix,frame_index=round(t*30),ankle_targets=stance,
        arm_actions={'right':{'action':'reach','phase':reach},
                     'left':{'action':'reach','phase':.68*ease((dt-.27)/.55)*(1-.82*ease((dt-1.2)/.9))}},
        allow_cut=round(t*30)==90,head_angle_deg=-7*turn-2*shoulder_pull,
        head_variant='shock' if dt>=.47 else None)
    if reach==1:
        wrist=matrix@np.append(actor.arms['right'][2],1)
        assert np.linalg.norm(wrist[:2]-grip)<.001
    return actor,grip

def handrail(im,grip):
    w,h=im.size
    x,y=grip
    pen=ImageDraw.Draw(im)
    width=max(2,round(w*.031))
    # Endpoint coincides with the closed hand. The stand feet sit on the same deck.
    for dx in (.055,.205):
        post=x+dx*w
        pen.line((post,y-dx*52,post-w*.010,y+h*.192),fill='#30232a',width=width)
        pen.line((post-w*.004,y,post-w*.013,y+h*.183),fill='#9d674b',width=max(1,width//4))
    end=(w+4,y-h*.038)
    pen.line((x,y,*end),fill='#2d2028',width=width+4)
    pen.line((x,y-2,*end),fill='#845234',width=width-4)
    pen.line((x,y-6,end[0],end[1]-5),fill='#c49060',width=max(2,width//5))

def frame(t,c):
    size=c.viewport.size
    w,h=size
    shot=next(q for q in c.plan['shots'] if q['start']<=t<q['end'])
    sid=shot['id']
    dt=t-shot['start']
    pos=ease(dt/(shot['end']-shot['start']))
    actors=[]
    if sid in ('crack','break'):
        im=collision(size,dt,sid)
    elif sid=='trap':
        im=plate('trapped',size,1.17-.16*pos,.44+.15*pos,.5)
        marker(im,'ANTARCTICA • 1915')
    elif sid=='alarm':
        im=plate('deck',size,1.04,.5,.51)
        shiver=ease((dt-.32)/.10)*(1-ease((dt-.48)/.33))
        if shiver:
            im=Image.blend(im,Image.new('RGBA',size,'#071c30'),.16*shiver)
        shade=Image.new('RGBA',size)
        d=ImageDraw.Draw(shade)
        d.ellipse((.34*w,.808*h,.64*w,.838*h),fill=(15,15,27,125))
        im.alpha_composite(shade.filter(ImageFilter.GaussianBlur(max(2,w*.016))))
        actor,grip=rail_action(t,c,w,h,dt)
        handrail(im,grip)
        actors.append(actor)
    elif sid in ('camp','survive'):
        im=plate('camp',size,1.12-.09*pos,.52,.49)
        actors.append(explorer(t,c,w,h,dt,sid))
    elif sid=='sink':
        im=collision(size,dt,'sink')
    elif sid=='voyage':
        im=plate('water',size,1.17,.50+.065*math.sin(dt*.95),.49)
        craft=boat(round(w*.64)).rotate(8*math.sin(dt*2.45),Image.Resampling.BICUBIC,expand=True)
        im.alpha_composite(craft,(round(w*(.38+.20*pos)-craft.width/2),round(h*(.63+.025*math.sin(dt*2.45))-craft.height)))
        marker(im,'800 MILES')
    else:
        clean=plate('wreck',size,1.08+.14*pos,.48+.045*pos,.53)
        dark=Image.blend(clean,Image.new('RGBA',size,'#020c16'),.94 if sid=='dark' else .69)
        if sid=='dark':
            im=dark
        else:
            mask=Image.new('L',size)
            pen=ImageDraw.Draw(mask)
            x=w*(.14+.61*pos if sid=='search' else .52)
            pen.ellipse((x-w*.37,h*.27,x+w*.37,h*.75),fill=235)
            im=Image.composite(clean,dark,mask.filter(ImageFilter.GaussianBlur(w*.16)))
            marker(im,'2022 • 3,008 M DOWN' if sid=='search' else 'ENDURANCE')
    flakes(im,t,sid in ('dark','search','name','sink'))
    return c.puppet_runtime.render_scene(size,actors,background=im,frame_index=round(t*30))

draft_frame=frame
